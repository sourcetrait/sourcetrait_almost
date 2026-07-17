//! The needle battery engine (the A-track instrument): spec-driven
//! passkey grids over a live model, single and multi-distractor modes,
//! per-cell results serializable for the ratchet artifacts.
//!
//! The method is the lastmost C3 instrument as library code: LCG word
//! filler seeded per target length, a 4-round calibration loop that
//! converges the chat-wrapped token count onto the target, greedy
//! decode, substring match. Multi mode plants two distractor needles
//! (the next keys in the spec pool, at offset depths) and asks only
//! the target - the lost-in-the-middle probe single mode cannot see.
use crate::*;

/// One passkey the spec can plant ({key}/{value} template slots).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NeedleKey {
    pub name: String,
    pub value: String,
}

/// The lastmost spec_v1.json shape, loaded verbatim.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NeedleSpec {
    pub depths: Vec<usize>,
    pub keys: Vec<NeedleKey>,
    pub filler_seed: u64,
    pub gen_tokens: usize,
    pub needle_template: String,
    pub question_template: String,
    pub words: Vec<String>,
}

impl NeedleSpec {
    pub fn load(path: &Path) -> LibAlmostResult<Self> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeedleMode {
    Single,
    /// Two distractors (the next keys cyclically) planted at
    /// (depth + 33)% and (depth + 67)%; the question asks the target.
    Multi,
}

impl NeedleMode {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Multi => "multi",
        }
    }
}

/// One battery cell's outcome. `found` is the target-value substring
/// match; `distractor_hits` records leaked distractor values (multi).
#[derive(Debug, Clone, serde::Serialize)]
pub struct NeedleCellResult {
    pub mode: String,
    pub target_length: usize,
    pub key: String,
    pub depth: usize,
    pub prompt_tokens: usize,
    pub found: bool,
    pub distractor_hits: Vec<String>,
    pub response: String,
}

/// A built cell prompt plus the distractor values planted in it.
pub(crate) struct CellPrompt {
    pub(crate) content: String,
    pub(crate) distractor_values: Vec<String>,
}

/// The spec's LCG: state = (state * 1103515245 + 12345) mod 2^31,
/// word = words[state mod pool]. Seeded filler_seed + target length.
pub(crate) fn lcg_words(words: &[String], count: usize, seed: u64) -> Vec<String> {
    let mut state = seed;
    let pool = words.len() as u64;
    (0..count)
        .map(|_| {
            state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % 2_147_483_648;
            words[(state % pool) as usize].clone()
        })
        .collect()
}

/// Join `words` with each plant's text inserted at its word index
/// (plants sorted ascending, indices strictly increasing).
pub(crate) fn splice(words: &[String], plants: &[(usize, &str)]) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    for (index, text) in plants {
        out.push_str(&words[cursor..*index].join(" "));
        out.push(' ');
        out.push_str(text);
        out.push(' ');
        cursor = *index;
    }
    out.push_str(&words[cursor..].join(" "));
    out
}

/// The (depth, key_index) plan for a cell: the target first, then the
/// mode's distractors at fixed depth offsets so cells stay
/// deterministic and internally comparable.
pub(crate) fn cell_plan(
    mode: NeedleMode,
    key_count: usize,
    key_index: usize,
    depth: usize,
) -> Vec<(usize, usize)> {
    match mode {
        NeedleMode::Single => vec![(depth, key_index)],
        NeedleMode::Multi => vec![
            (depth, key_index),
            ((depth + 33) % 100, (key_index + 1) % key_count),
            ((depth + 67) % 100, (key_index + 2) % key_count),
        ],
    }
}

/// Cell-prompt builder over one spec + tokenizer (calibration needs
/// live token counts).
pub(crate) struct NeedleRig<'a> {
    pub(crate) spec: &'a NeedleSpec,
    pub(crate) tokenizer: &'a tokenizers::Tokenizer,
}

impl NeedleRig<'_> {
    fn encoded_len(&self, text: &str) -> LibAlmostResult<usize> {
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => Ok(encoding.get_ids().len()),
            Err(e) => snafu::whatever!("needle prompt encode failed: {e}"),
        }
    }

    fn render_needle(&self, key: &NeedleKey) -> String {
        self.spec
            .needle_template
            .replace("{key}", &key.name)
            .replace("{value}", &key.value)
    }

    /// Build one cell's calibrated content (the 4-round loop from the
    /// C3 method, generalized to N plants).
    pub(crate) fn cell_prompt(
        &self,
        mode: NeedleMode,
        target: usize,
        key_index: usize,
        depth: usize,
    ) -> LibAlmostResult<CellPrompt> {
        let spec = self.spec;
        let plan = cell_plan(mode, spec.keys.len(), key_index, depth);
        let needles: Vec<String> = plan
            .iter()
            .map(|(_, key)| self.render_needle(&spec.keys[*key]))
            .collect();
        let question = spec
            .question_template
            .replace("{key}", &spec.keys[key_index].name);

        let base_words = lcg_words(
            &spec.words,
            (target * 2).max(4096),
            spec.filler_seed + target as u64,
        );
        let probe = base_words[..2000].join(" ");
        let ratio = self.encoded_len(&probe)? as f64 / 2000.0;
        let overhead = self.encoded_len(&chat_wrap(&format!(
            "{}\n\n{question}",
            needles.join("\n\n")
        )))?;

        let mut count = (((target.saturating_sub(overhead)) as f64 / ratio) as i64).max(64);
        let mut content = String::new();
        for _ in 0..4 {
            let words = &base_words[..(count as usize).min(base_words.len())];
            // Depth -> word index; equal indices bump forward so every
            // plant lands (strictly increasing for the splice).
            let mut indexed: Vec<(usize, &str)> = plan
                .iter()
                .zip(&needles)
                .map(|((plant_depth, _), text)| {
                    (
                        (words.len() * plant_depth / 100).clamp(1, words.len() - 1),
                        text.as_str(),
                    )
                })
                .collect();
            indexed.sort_by_key(|(index, _)| *index);
            for position in 1..indexed.len() {
                if indexed[position].0 <= indexed[position - 1].0 {
                    indexed[position].0 = (indexed[position - 1].0 + 1).min(words.len() - 1);
                }
            }
            content = format!("{}\n\n{question}", splice(words, &indexed));
            let actual = self.encoded_len(&chat_wrap(&content))? as i64;
            if (actual - target as i64).abs() <= 8.max(target as i64 / 200) {
                break;
            }
            count = (count + ((target as i64 - actual) as f64 / ratio) as i64).max(64);
        }
        Ok(CellPrompt {
            content,
            distractor_values: plan[1..]
                .iter()
                .map(|(_, key)| spec.keys[*key].value.clone())
                .collect(),
        })
    }
}

/// Run one mode's full grid (keys x depths per length) greedy, one
/// generation per cell over the model's carried caches.
pub fn run_needle_cells(
    model: &mut OlmoHybrid,
    tokenizer: &tokenizers::Tokenizer,
    spec: &NeedleSpec,
    mode: NeedleMode,
    lengths: &[usize],
) -> LibAlmostResult<Vec<NeedleCellResult>> {
    let rig = NeedleRig { spec, tokenizer };
    let mut cells = Vec::new();
    for &target in lengths {
        for key_index in 0..spec.keys.len() {
            for &depth in &spec.depths {
                let prompt = rig.cell_prompt(mode, target, key_index, depth)?;
                let options = GenerateOptions::greedy(spec.gen_tokens);
                let mut generation = model.generate(tokenizer, &prompt.content, &options)?;
                let mut response = String::new();
                for step in generation.by_ref() {
                    response.push_str(&step?.chunk);
                }
                let report = generation.finish();
                response.push_str(&report.rest);
                let target_key = &spec.keys[key_index];
                cells.push(NeedleCellResult {
                    mode: mode.label().to_string(),
                    target_length: target,
                    key: target_key.name.clone(),
                    depth,
                    prompt_tokens: report.prompt_token_count,
                    found: response.contains(&target_key.value),
                    distractor_hits: prompt
                        .distractor_values
                        .iter()
                        .filter(|value| response.contains(value.as_str()))
                        .cloned()
                        .collect(),
                    response,
                });
            }
        }
    }
    Ok(cells)
}
