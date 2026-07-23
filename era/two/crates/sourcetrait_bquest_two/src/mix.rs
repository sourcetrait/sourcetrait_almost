//! The training-mix pipeline's document stage: render corpus trees
//! into dolma-FIELD documents (one FILE per document, text = the
//! verbatim bytes, identity in metadata only - the lineage
//! presentation; bquest understood 03). Artifacts are whole-value
//! .nuon tables, one per spec row - document text carries raw
//! newlines, which the NUON-LINES form's single-line guard rightly
//! refuses, so the lines form is reserved for logs.
use crate::*;

/// The render spec (`--spec`): one row per corpus tree.
pub(crate) const MIX_SPEC_TYPEDEF: &str = "table<name: string, corpus_dir: string, \
     repo: string, license: string, kind: string>";

/// One rendered document (the dolma field shape; kind rides metadata
/// so the pack stage can gate FIM to code documents).
pub(crate) const MIX_DOCUMENT_TYPEDEF: &str = "record<id: string, text: string, \
     source: string, added: string, created: string, \
     metadata: record<repo: string, path: string, language: string, \
     license: string, kind: string>>";

struct MixSpecRow {
    name: String,
    corpus_dir: PathBuf,
    repo: String,
    license: String,
    kind: String,
}

fn read_spec(value: &lib::nu::Value) -> BquestResult<Vec<MixSpecRow>> {
    let rows = match value.as_list() {
        Ok(list) => list,
        Err(e) => snafu::whatever!("the mix spec must be a table: {e}"),
    };
    let mut spec_rows = Vec::with_capacity(rows.len());
    for row in rows {
        let record = match row.as_record() {
            Ok(record) => record,
            Err(e) => snafu::whatever!("mix spec row: {e}"),
        };
        spec_rows.push(MixSpecRow {
            name: field_str(record, "name")?,
            corpus_dir: PathBuf::from(field_str(record, "corpus_dir")?),
            repo: field_str(record, "repo")?,
            license: field_str(record, "license")?,
            kind: field_str(record, "kind")?,
        });
    }
    Ok(spec_rows)
}

/// The metadata language token, from the file extension (lowercase;
/// unmapped extensions carry verbatim, extensionless files read
/// "text").
fn language_token(path: &Path) -> String {
    let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
        return String::from("text");
    };
    match extension.to_ascii_lowercase().as_str() {
        "rs" => String::from("rust"),
        "md" => String::from("markdown"),
        other => other.to_string(),
    }
}

/// Render one corpus tree into document rows, walk sorted for
/// deterministic output.
fn render_tree(spec: &MixSpecRow, stamp: &str) -> BquestResult<Vec<lib::nu::Value>> {
    snafu::ensure_whatever!(
        spec.corpus_dir.is_dir(),
        "corpus_dir {} is not a directory",
        spec.corpus_dir.display()
    );
    let mut files: Vec<PathBuf> = Vec::new();
    let mut pending: Vec<PathBuf> = vec![spec.corpus_dir.clone()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();

    let mut documents = Vec::with_capacity(files.len());
    for path in &files {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => snafu::whatever!("read {} failed (corpus is utf-8): {e}", path.display()),
        };
        let relative = match path.strip_prefix(&spec.corpus_dir) {
            Ok(relative) => relative.display().to_string(),
            Err(_) => path.display().to_string(),
        };
        documents.push(lib::nu::Value::record(
            lib::nu::record! {
                "id" => v_str(&format!("{}/{relative}", spec.repo)),
                "text" => v_str(&text),
                "source" => v_str(&spec.repo),
                "added" => v_str(stamp),
                "created" => v_str(stamp),
                "metadata" => lib::nu::Value::record(
                    lib::nu::record! {
                        "repo" => v_str(&spec.repo),
                        "path" => v_str(&relative),
                        "language" => v_str(&language_token(path)),
                        "license" => v_str(&spec.license),
                        "kind" => v_str(&spec.kind),
                    },
                    span(),
                ),
            },
            span(),
        ));
    }
    Ok(documents)
}

// The FIM family is consumed by the pack stage (next in CptLoop);
// the allows retire with it.
/// FIM sentinel spellings (the pipe-wrapped forms the tokenizer
/// carries as single special tokens; the lineage's exact bytes).
#[allow(dead_code)]
pub(crate) const FIM_PREFIX: &str = "<|fim_prefix|>";
#[allow(dead_code)]
pub(crate) const FIM_MIDDLE: &str = "<|fim_middle|>";
#[allow(dead_code)]
pub(crate) const FIM_SUFFIX: &str = "<|fim_suffix|>";
/// The lineage FIM rate (per document) and the PSM/SPM split.
#[allow(dead_code)]
pub(crate) const FIM_RATE: f64 = 0.5;

/// SplitMix64: the deterministic seed-expanding rng (the era-one
/// packing precedent; no external rng dependency).
#[allow(dead_code)]
pub(crate) struct SplitMix64 {
    state: u64,
}

#[allow(dead_code)]
impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D049BB133111EB);
        mixed ^ (mixed >> 31)
    }

    /// Uniform in [0, 1).
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [0, bound) (bound > 0).
    pub(crate) fn next_below(&mut self, bound: usize) -> usize {
        (self.next_unit() * bound as f64) as usize % bound
    }
}

/// The lineage FIM transform: at FIM_RATE per document pick two char
/// break points, then render 50/50 PSM
/// (prefix-sentinel prefix, suffix-sentinel suffix, middle-sentinel
/// middle) vs SPM (the suffix span leads); otherwise the text passes
/// through untouched. Char-boundary safe; the three spans always
/// reassemble the original text.
#[allow(dead_code)]
pub(crate) fn fim_transform(text: &str, rng: &mut SplitMix64) -> String {
    if rng.next_unit() >= FIM_RATE {
        return text.to_string();
    }
    let char_count = text.chars().count();
    if char_count < 2 {
        return text.to_string();
    }
    let mut first_break = rng.next_below(char_count + 1);
    let mut second_break = rng.next_below(char_count + 1);
    if first_break > second_break {
        std::mem::swap(&mut first_break, &mut second_break);
    }
    let byte_of = |char_index: usize| -> usize {
        text.char_indices()
            .nth(char_index)
            .map(|(byte, _)| byte)
            .unwrap_or(text.len())
    };
    let first_byte = byte_of(first_break);
    let second_byte = byte_of(second_break);
    let prefix = &text[..first_byte];
    let middle = &text[first_byte..second_byte];
    let suffix = &text[second_byte..];
    if rng.next_unit() < 0.5 {
        format!("{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}")
    } else {
        format!("{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}")
    }
}

/// One document's pack-relevant fields (text + the FIM gate).
#[allow(dead_code)]
pub(crate) struct PackDocument {
    pub(crate) text: String,
    pub(crate) code: bool,
}

/// Tokenize documents in order into one EOS-joined id stream: FIM
/// (when armed) transforms CODE documents pre-tokenization - the
/// sentinels are added tokens, so they land as single ids - and
/// every document ends with the EOS id.
#[allow(dead_code)]
pub(crate) fn tokenize_documents(
    tokenizer: &tokenizers::Tokenizer,
    documents: &[PackDocument],
    fim: bool,
    rng: &mut SplitMix64,
) -> BquestResult<Vec<u32>> {
    let Some(eos_id) = tokenizer.token_to_id("<|endoftext|>") else {
        snafu::whatever!("the tokenizer carries no <|endoftext|>");
    };
    let mut stream: Vec<u32> = Vec::new();
    for document in documents {
        let presented = if fim && document.code {
            fim_transform(&document.text, rng)
        } else {
            document.text.clone()
        };
        let encoding = match tokenizer.encode(presented.as_str(), false) {
            Ok(encoding) => encoding,
            Err(e) => snafu::whatever!("document encode failed: {e}"),
        };
        stream.extend_from_slice(encoding.get_ids());
        stream.push(eos_id);
    }
    Ok(stream)
}

/// Cut the joined stream into (seq_len + 1) chunks and shuffle them
/// (Fisher-Yates over the chunk order, SplitMix64-seeded); the
/// ragged tail is dropped and reported as the second return.
#[allow(dead_code)]
pub(crate) fn chunk_and_shuffle(
    stream: &[u32],
    seq_len: usize,
    rng: &mut SplitMix64,
) -> (Vec<Vec<u32>>, usize) {
    let width = seq_len + 1;
    let chunk_count = stream.len() / width;
    let dropped_tail = stream.len() - chunk_count * width;
    let mut chunks: Vec<Vec<u32>> = (0..chunk_count)
        .map(|index| stream[index * width..(index + 1) * width].to_vec())
        .collect();
    for index in (1..chunks.len()).rev() {
        let swap_with = rng.next_below(index + 1);
        chunks.swap(index, swap_with);
    }
    (chunks, dropped_tail)
}

/// `bquest mix render`: corpus trees (per the spec) -> one
/// whole-value documents_<name>.nuon table per spec row, rows
/// conform-validated against the document typedef before write.
pub(crate) fn mix_render(args: &MixRenderArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let spec_value = lib::nu::load_value(&args.spec)?;
    lib::nu::conform(&spec_value, &lib::nu::parse_typedef(MIX_SPEC_TYPEDEF)?)?;
    let spec_rows = read_spec(&spec_value)?;
    let stamp = args.stamp.clone().unwrap_or_default();
    let document_type = lib::nu::parse_typedef(MIX_DOCUMENT_TYPEDEF)?;

    let mut summary_rows: Vec<lib::nu::Value> = Vec::new();
    for spec_row in &spec_rows {
        let documents = render_tree(spec_row, &stamp)?;
        for document in &documents {
            lib::nu::conform(document, &document_type)?;
        }
        let text_bytes: usize = documents
            .iter()
            .map(|document| {
                document
                    .as_record()
                    .ok()
                    .and_then(|record| record.get("text"))
                    .and_then(|text| text.as_str().ok())
                    .map(str::len)
                    .unwrap_or(0)
            })
            .sum();
        let path = args.out.join(format!("documents_{}.nuon", spec_row.name));
        let count = documents.len();
        lib::nu::save_value(&path, &lib::nu::Value::list(documents, span()))?;
        eprintln!(
            "mix render: {} - {count} documents, {text_bytes} text bytes -> {}",
            spec_row.name,
            path.display()
        );
        summary_rows.push(lib::nu::Value::record(
            lib::nu::record! {
                "name" => v_str(&spec_row.name),
                "documents" => v_int(count as i64),
                "text_bytes" => v_int(text_bytes as i64),
            },
            span(),
        ));
    }
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "rendered" => lib::nu::Value::list(summary_rows, span()),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
