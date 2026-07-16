use crate::*;

/// The packed training stream: fixed-length id chunks (seq_len + 1 each,
/// inputs overlap targets by one), shuffled deterministically.
pub(crate) struct Packed {
    pub(crate) chunks: Vec<Vec<u32>>,
    pub(crate) total_tokens: usize,
    pub(crate) documents: usize,
}

/// SplitMix64: deterministic, dependency-free shuffling (independent copy;
/// the oracle stays dependency-light on purpose).
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }
}

fn wanted(path: &Path, extensions: &[String]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| extensions.iter().any(|wanted| wanted == ext))
}

fn walk(
    dir: &Path,
    excludes: &[String],
    extensions: &[String],
    out: &mut Vec<PathBuf>,
) -> HeatResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name.starts_with('.') || excludes.iter().any(|e| e.as_str() == name) {
                continue;
            }
            walk(&path, excludes, extensions, out)?;
        } else if wanted(&path, extensions) {
            out.push(path);
        }
    }
    Ok(())
}

/// Walk the corpus roots for the given extensions, tokenize each document
/// (under a `### FILE:` path-header line when with_headers - the domain
/// convention; general admixture text stays raw), join with the eos id,
/// and pack into shuffled (seq_len + 1) chunks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_packed(
    roots: &[PathBuf],
    excludes: &[String],
    extensions: &[String],
    with_headers: bool,
    tokenizer: &tokenizers::Tokenizer,
    seq_len: usize,
    eos: u32,
    seed: u64,
) -> HeatResult<Packed> {
    let mut stream: Vec<u32> = Vec::new();
    let mut documents = 0usize;
    for root in roots {
        let mut files: Vec<PathBuf> = Vec::new();
        walk(root, excludes, extensions, &mut files)?;
        files.sort();
        for file in files {
            let Ok(content) = std::fs::read_to_string(&file) else {
                continue;
            };
            let text = if with_headers {
                let relative = file.strip_prefix(root).unwrap_or(&file);
                format!("### FILE: {}\n{content}", relative.display())
            } else {
                content
            };
            let encoding = match tokenizer.encode(text, false) {
                Ok(encoding) => encoding,
                Err(error) => snafu::whatever!("tokenization of {} failed: {error}", file.display()),
            };
            stream.extend_from_slice(encoding.get_ids());
            stream.push(eos);
            documents += 1;
        }
    }
    snafu::ensure_whatever!(
        stream.len() > seq_len,
        "corpus tokenized to {} tokens - not enough for one {seq_len}-token chunk",
        stream.len()
    );

    let mut chunks: Vec<Vec<u32>> = Vec::new();
    let mut start = 0usize;
    while start + seq_len < stream.len() {
        chunks.push(stream[start..start + seq_len + 1].to_vec());
        start += seq_len;
    }

    // Fisher-Yates under the deterministic generator.
    let mut rng = SplitMix64(seed);
    for index in (1..chunks.len()).rev() {
        let pick = (rng.next_u64() % (index as u64 + 1)) as usize;
        chunks.swap(index, pick);
    }

    Ok(Packed {
        total_tokens: stream.len(),
        chunks,
        documents,
    })
}

/// Interleave general-admixture chunks into the domain schedule at
/// `ratio` general chunks per domain chunk (credit-accumulated, so
/// fractional ratios pace evenly). General chunks cycle when the domain
/// pass outlasts them; both inputs arrive pre-shuffled.
pub(crate) fn compose_mixed(domain: Packed, general: Packed, ratio: f64) -> Packed {
    if general.chunks.is_empty() || ratio <= 0.0 {
        return domain;
    }
    let mut chunks = Vec::with_capacity(
        domain.chunks.len() + (domain.chunks.len() as f64 * ratio) as usize + 1,
    );
    let mut general_cursor = 0usize;
    let mut credit = 0f64;
    for domain_chunk in domain.chunks {
        chunks.push(domain_chunk);
        credit += ratio;
        while credit >= 1.0 {
            chunks.push(general.chunks[general_cursor % general.chunks.len()].clone());
            general_cursor += 1;
            credit -= 1.0;
        }
    }
    Packed {
        total_tokens: domain.total_tokens + general.total_tokens,
        documents: domain.documents + general.documents,
        chunks,
    }
}
