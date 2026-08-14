//! The ImagineQuestMatrix build: the hand-built associative embedding.
use crate::*;

use crate::bucket::BucketTable;
use crate::census::read_admitted;
use crate::lexer::KEYWORD_PAGE_SIZE;
use crate::lexer::KEYWORD_REPETITION;
use crate::ucd::CharacterTable;

/// The attention/GDN head count (TheUser-ruled); width is arithmetic.
pub(crate) const MATRIX_HEADS: usize = 32;
/// The base init scale for every seeded row.
const ROW_SIGMA: f64 = 0.02;
/// The identity offset a composed row carries beside its components.
const JITTER_SIGMA: f64 = 0.004;
/// The artifact's embedding tensor, as the model tree names it.
pub(crate) const EMBED_TENSOR: &str = "model.embed_tokens.weight";

/// The layered id space the matrix rows mirror, offset for offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixLayout {
    pub(crate) character_offset: usize,
    pub(crate) keyboard_offset: usize,
    pub(crate) dictionary_offset: usize,
    pub(crate) reserve_offset: usize,
    pub(crate) vocab_size: usize,
    pub(crate) hidden: usize,
}

impl MatrixLayout {
    pub(crate) fn new(
        table: &CharacterTable,
        buckets: &BucketTable,
        admitted: usize,
        reserve: usize,
        hidden: usize,
    ) -> Self {
        let character_offset = KEYWORD_PAGE_SIZE as usize;
        let keyboard_offset = character_offset + table.assigned_count();
        let dictionary_offset = keyboard_offset + buckets.count();
        let reserve_offset = dictionary_offset + admitted;
        Self {
            character_offset,
            keyboard_offset,
            dictionary_offset,
            reserve_offset,
            vocab_size: reserve_offset + reserve,
            hidden,
        }
    }
}

/// The built matrix: the layout plus vocab x hidden f32 rows.
pub(crate) struct MatrixBuild {
    pub(crate) layout: MatrixLayout,
    pub(crate) rows: Vec<f32>,
}

/// One standard-normal draw, Box-Muller over SplitMix64.
fn normal_draw(rng: &mut llm::SplitMix64) -> f64 {
    let mut first_unit = rng.next_unit();
    if first_unit <= f64::MIN_POSITIVE {
        first_unit = f64::MIN_POSITIVE;
    }
    let second_unit = rng.next_unit();
    (-2.0 * first_unit.ln()).sqrt() * (2.0 * std::f64::consts::PI * second_unit).cos()
}

/// Row-kind tags keeping per-row seeds disjoint across the layers.
const SEED_KEYWORD: u64 = 1 << 56;
const SEED_CHARACTER: u64 = 2 << 56;
const SEED_KEYBOARD: u64 = 3 << 56;
const SEED_RESERVE: u64 = 4 << 56;

/// A word's stable seed (FNV-1a over its bytes), independent of its
/// admitted position, so a shared word seeds alike across corpora.
pub(crate) fn word_seed(word: &str) -> u64 {
    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    for byte in word.as_bytes() {
        acc ^= *byte as u64;
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    acc
}

impl MatrixBuild {
    fn row_mut(&mut self, index: usize) -> &mut [f32] {
        let hidden = self.layout.hidden;
        &mut self.rows[index * hidden..(index + 1) * hidden]
    }

    fn row(&self, index: usize) -> &[f32] {
        let hidden = self.layout.hidden;
        &self.rows[index * hidden..(index + 1) * hidden]
    }

    /// Fill one row with seeded gaussians at `sigma`.
    fn seed_row(&mut self, index: usize, seed: u64, sigma: f64) {
        let mut rng = llm::SplitMix64::new(seed);
        for value in self.row_mut(index) {
            *value = (normal_draw(&mut rng) * sigma) as f32;
        }
    }

    /// Add seeded gaussians at `sigma` onto an existing row.
    fn jitter_row(&mut self, index: usize, seed: u64, sigma: f64) {
        let mut rng = llm::SplitMix64::new(seed);
        for value in self.row_mut(index) {
            *value += (normal_draw(&mut rng) * sigma) as f32;
        }
    }

    /// Compose a row from component rows: sum over sqrt(k), which
    /// keeps the composed variance at the components' own scale.
    fn compose_row(&mut self, index: usize, components: &[usize]) {
        let hidden = self.layout.hidden;
        let scale = 1.0 / (components.len().max(1) as f64).sqrt();
        let mut composed = vec![0f64; hidden];
        for &component in components {
            for (slot, value) in composed.iter_mut().zip(self.row(component)) {
                *slot += *value as f64;
            }
        }
        for (slot, value) in self.row_mut(index).iter_mut().zip(&composed) {
            *slot = (*value * scale) as f32;
        }
    }
}

/// A character's absolute row index, or a refusal.
fn character_index(
    table: &CharacterTable,
    layout: &MatrixLayout,
    c: char,
) -> BiquestResult<usize> {
    match table.index_of(c as u32) {
        Some(index) => Ok(layout.character_offset + index as usize),
        None => snafu::whatever!("no character row for U+{:04X}", c as u32),
    }
}

/// The build's scalar knobs, gathered into one argument.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatrixSpec {
    pub(crate) hidden: usize,
    pub(crate) reserve: usize,
    pub(crate) seed: u64,
    pub(crate) sigma: f64,
    pub(crate) jitter_sigma: f64,
}

/// Build the full associative matrix in memory, deterministically.
///
/// Keyword and character rows seed as plain gaussians (characters by
/// code point, so shared rows survive a Unicode bump). A keyboard row
/// composes from exactly its declared associations - its character,
/// its count digit, and the repetition AbstractConceptMarker row -
/// and a word row from its component character rows; both carry a
/// small identity jitter so equal-multiset spellings start distinct.
/// Reserve rows seed as plain gaussians awaiting a mint.
pub(crate) fn build_matrix(
    table: &CharacterTable,
    buckets: &BucketTable,
    admitted: &[String],
    spec: MatrixSpec,
) -> BiquestResult<MatrixBuild> {
    let MatrixSpec { hidden, reserve, seed, sigma, jitter_sigma } = spec;
    let layout = MatrixLayout::new(table, buckets, admitted.len(), reserve, hidden);
    let mut build = MatrixBuild {
        layout,
        rows: vec![0f32; layout.vocab_size * hidden],
    };

    for id in 0..layout.character_offset {
        build.seed_row(id, seed ^ SEED_KEYWORD ^ id as u64, sigma);
    }
    for (index, row) in table.rows.iter().enumerate() {
        build.seed_row(
            layout.character_offset + index,
            seed ^ SEED_CHARACTER ^ row.code_point as u64,
            sigma,
        );
    }

    let repetition_row = KEYWORD_REPETITION as usize;
    for index in 0..buckets.count() {
        let entry = buckets.entry(index);
        let digit = char::from(b'0' + entry.count);
        let components = [
            character_index(table, &layout, entry.symbol)?,
            character_index(table, &layout, digit)?,
            repetition_row,
        ];
        let row = layout.keyboard_offset + index;
        build.compose_row(row, &components);
        let identity = SEED_KEYBOARD ^ ((entry.symbol as u64) << 8) ^ entry.count as u64;
        build.jitter_row(row, seed ^ identity, jitter_sigma);
    }

    for (position, word) in admitted.iter().enumerate() {
        let mut components = Vec::with_capacity(word.chars().count());
        for c in word.chars() {
            components.push(character_index(table, &layout, c)?);
        }
        snafu::ensure_whatever!(
            !components.is_empty(),
            "admitted word {position} is empty"
        );
        let row = layout.dictionary_offset + position;
        build.compose_row(row, &components);
        build.jitter_row(row, seed ^ word_seed(word), jitter_sigma);
    }

    for index in 0..(layout.vocab_size - layout.reserve_offset) {
        build.seed_row(
            layout.reserve_offset + index,
            seed ^ SEED_RESERVE ^ index as u64,
            sigma,
        );
    }
    Ok(build)
}

/// `biquest matrix build`: the embedding artifact from the embedded
/// layers plus an admitted wordlist; bf16 safetensors, tied by design.
pub(crate) fn matrix_build(args: &MatrixBuildArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let admitted = read_admitted(&args.admitted)?;
    let hidden = MATRIX_HEADS * args.head_dim;
    let build = build_matrix(
        &table,
        &buckets,
        &admitted,
        MatrixSpec {
            hidden,
            reserve: args.reserve,
            seed: args.seed,
            sigma: ROW_SIGMA,
            jitter_sigma: JITTER_SIGMA,
        },
    )?;
    let layout = build.layout;

    let bytes: Vec<u8> = build
        .rows
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
        .collect();
    let view = match safetensors::tensor::TensorView::new(
        safetensors::Dtype::BF16,
        vec![layout.vocab_size, hidden],
        &bytes,
    ) {
        Ok(view) => view,
        Err(e) => snafu::whatever!("embedding view failed: {e}"),
    };
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut meta = |key: &str, value: String| {
        metadata.insert(String::from(key), value);
    };
    meta("version", String::from("1"));
    meta("kind", String::from("imagine_quest_matrix"));
    meta("tie_word_embeddings", String::from("true"));
    meta("unicode_assigned", table.assigned_count().to_string());
    meta("character_offset", layout.character_offset.to_string());
    meta("keyboard_offset", layout.keyboard_offset.to_string());
    meta("dictionary_offset", layout.dictionary_offset.to_string());
    meta("reserve_offset", layout.reserve_offset.to_string());
    meta("vocab_size", layout.vocab_size.to_string());
    meta("hidden_size", hidden.to_string());
    meta("num_heads", MATRIX_HEADS.to_string());
    meta("head_dim", args.head_dim.to_string());
    meta("admitted_words", admitted.len().to_string());
    meta("reserve_rows", args.reserve.to_string());
    meta("seed", args.seed.to_string());
    meta("sigma", ROW_SIGMA.to_string());
    meta("jitter_sigma", JITTER_SIGMA.to_string());
    meta("tool_version", String::from(env!("CARGO_PKG_VERSION")));
    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    match safetensors::serialize_to_file(
        vec![(String::from(EMBED_TENSOR), view)],
        Some(metadata),
        &args.out,
    ) {
        Ok(()) => {}
        Err(e) => snafu::whatever!("matrix write failed ({}): {e}", args.out.display()),
    }

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "admitted" => v_str(&args.admitted.display().to_string()),
            "unicode_assigned" => v_int(table.assigned_count() as i64),
            "keyword_rows" => v_int(layout.character_offset as i64),
            "character_rows" => v_int((layout.keyboard_offset - layout.character_offset) as i64),
            "keyboard_rows" => v_int((layout.dictionary_offset - layout.keyboard_offset) as i64),
            "dictionary_rows" => v_int((layout.reserve_offset - layout.dictionary_offset) as i64),
            "reserve_rows" => v_int((layout.vocab_size - layout.reserve_offset) as i64),
            "vocab_size" => v_int(layout.vocab_size as i64),
            "hidden_size" => v_int(hidden as i64),
            "num_heads" => v_int(MATRIX_HEADS as i64),
            "head_dim" => v_int(args.head_dim as i64),
            "seed" => v_int(args.seed as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "vocab_size" => v_int(layout.vocab_size as i64),
            "hidden_size" => v_int(hidden as i64),
            "dictionary_rows" => v_int((layout.reserve_offset - layout.dictionary_offset) as i64),
            "bytes" => v_int(bytes.len() as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
