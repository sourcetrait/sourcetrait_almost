//! The training-mix pipeline's document stage: render corpus trees
//! into dolma-FIELD documents (one FILE per document, text = the
//! verbatim bytes, identity in metadata only - the lineage
//! presentation; bquest understood 03). Artifacts are whole-value
//! .nuon tables, one per spec row - document text carries raw
//! newlines, which the NUON-LINES form's single-line guard rightly
//! refuses, so the lines form is reserved for logs.
use crate::*;

use std::io::BufRead;

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

/// `bquest mix pack`: document tables (in the given mix order) ->
/// one shuffled packed-chunks artifact - "chunks" u32
/// [n, seq_len + 1] safetensors plus a .nuon provenance sidecar.
/// The model dir resolves through the global -c config (the
/// tokenizer is the checkpoint's own).
pub(crate) fn mix_pack(cli: &Cli, args: &MixPackArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let config = lib::LibConfig::load_from_dir(cli.dir.as_ref(), cli.config.as_deref())?;
    let tokenizer = lib::load_tokenizer(&config.model_dir())?;
    lib::verify_token_map(&tokenizer)?;
    let document_type = lib::nu::parse_typedef(MIX_DOCUMENT_TYPEDEF)?;

    let mut pack_documents: Vec<PackDocument> = Vec::new();
    for path in &args.documents {
        let table = lib::nu::load_value(path)?;
        let rows = match table.as_list() {
            Ok(rows) => rows,
            Err(e) => snafu::whatever!("{}: not a document table: {e}", path.display()),
        };
        for row in rows {
            lib::nu::conform(row, &document_type)?;
            let record = match row.as_record() {
                Ok(record) => record,
                Err(e) => snafu::whatever!("document row: {e}"),
            };
            let Some(metadata_value) = record.get("metadata") else {
                snafu::whatever!("document metadata is missing");
            };
            let metadata = match metadata_value.as_record() {
                Ok(metadata) => metadata,
                Err(e) => snafu::whatever!("document metadata is not a record: {e}"),
            };
            pack_documents.push(PackDocument {
                text: field_str(record, "text")?,
                code: field_str(metadata, "kind")? == "code",
            });
        }
    }

    let mut rng = SplitMix64::new(args.seed);
    let stream = tokenize_documents(&tokenizer, &pack_documents, args.fim, &mut rng)?;
    let (chunks, dropped_tail) = chunk_and_shuffle(&stream, args.seq_len, &mut rng);
    snafu::ensure_whatever!(
        !chunks.is_empty(),
        "the stream ({} tokens) is shorter than one chunk (seq_len {})",
        stream.len(),
        args.seq_len
    );

    let width = args.seq_len + 1;
    let mut chunk_bytes: Vec<u8> = Vec::with_capacity(chunks.len() * width * 4);
    for chunk in &chunks {
        for id in chunk {
            chunk_bytes.extend_from_slice(&id.to_le_bytes());
        }
    }
    let view = match safetensors::tensor::TensorView::new(
        safetensors::Dtype::U32,
        vec![chunks.len(), width],
        &chunk_bytes,
    ) {
        Ok(view) => view,
        Err(e) => snafu::whatever!("chunks view failed: {e}"),
    };
    if let Some(parent) = args.out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    match safetensors::serialize_to_file(
        vec![(String::from("chunks"), view)],
        None,
        &args.out,
    ) {
        Ok(()) => {}
        Err(e) => snafu::whatever!("chunk artifact write failed: {e}"),
    }

    let provenance_path = args.out.with_extension("nuon");
    let provenance = lib::nu::Value::record(
        lib::nu::record! {
            "documents" => lib::nu::Value::list(
                args.documents
                    .iter()
                    .map(|p| v_str(&p.display().to_string()))
                    .collect(),
                span(),
            ),
            "document_count" => v_int(pack_documents.len() as i64),
            "seq_len" => v_int(args.seq_len as i64),
            "seed" => v_int(args.seed as i64),
            "fim" => v_bool(args.fim),
            "total_tokens" => v_int(stream.len() as i64),
            "chunks" => v_int(chunks.len() as i64),
            "dropped_tail" => v_int(dropped_tail as i64),
            "bquest_version" => v_str(env!("CARGO_PKG_VERSION")),
        },
        span(),
    );
    lib::nu::save_value(&provenance_path, &provenance)?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "chunks" => v_int(chunks.len() as i64),
            "total_tokens" => v_int(stream.len() as i64),
            "dropped_tail" => v_int(dropped_tail as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// `bquest mix sample`: document tables -> one sampled table. One
/// seeded Fisher-Yates shuffle across the UNION of all input rows,
/// taken in shuffled order until the text-byte budget is crossed
/// (the final document overshoots; the overshoot is visible in the
/// reported text_bytes). The admixture-leg sampler: deterministic
/// per seed, provenance beside the artifact.
pub(crate) fn mix_sample(args: &MixSampleArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let document_type = lib::nu::parse_typedef(MIX_DOCUMENT_TYPEDEF)?;
    let mut rows: Vec<lib::nu::Value> = Vec::new();
    for path in &args.documents {
        let table = lib::nu::load_value(path)?;
        let list = match table.as_list() {
            Ok(list) => list,
            Err(e) => snafu::whatever!("{}: not a document table: {e}", path.display()),
        };
        for row in list {
            lib::nu::conform(row, &document_type)?;
            rows.push(row.clone());
        }
    }
    snafu::ensure_whatever!(!rows.is_empty(), "no documents to sample");

    let mut rng = SplitMix64::new(args.seed);
    for index in (1..rows.len()).rev() {
        let swap_with = rng.next_below(index + 1);
        rows.swap(index, swap_with);
    }

    let total_available = rows.len();
    let mut taken: Vec<lib::nu::Value> = Vec::new();
    let mut text_bytes = 0usize;
    for row in rows {
        if text_bytes >= args.budget_bytes {
            break;
        }
        let row_bytes = row
            .as_record()
            .ok()
            .and_then(|record| record.get("text"))
            .and_then(|text| text.as_str().ok())
            .map(str::len)
            .unwrap_or(0);
        text_bytes += row_bytes;
        taken.push(row);
    }
    let document_count = taken.len();
    lib::nu::save_value(&args.out, &lib::nu::Value::list(taken, span()))?;

    let provenance = lib::nu::Value::record(
        lib::nu::record! {
            "documents" => lib::nu::Value::list(
                args.documents
                    .iter()
                    .map(|p| v_str(&p.display().to_string()))
                    .collect(),
                span(),
            ),
            "budget_bytes" => v_int(args.budget_bytes as i64),
            "seed" => v_int(args.seed as i64),
            "documents_available" => v_int(total_available as i64),
            "documents_taken" => v_int(document_count as i64),
            "text_bytes" => v_int(text_bytes as i64),
            "bquest_version" => v_str(env!("CARGO_PKG_VERSION")),
        },
        span(),
    );
    lib::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "documents_taken" => v_int(document_count as i64),
            "documents_available" => v_int(total_available as i64),
            "text_bytes" => v_int(text_bytes as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
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

/// `bquest mix rip`: one zstd-compressed dolma JSONL shard -> a
/// document table, taking documents until a text-byte budget is
/// crossed. The their-side replay's ingest, in the same document
/// shape `mix render` produces for our own corpus trees, so both
/// sides feed `mix sample` and `mix pack` unchanged.
///
/// ## DEV
/// THE DECODER LIVES HERE because the box ships no zstd binary and
/// nushell reads none, so the harness previously drove a throwaway
/// Rust bin built per session and lost to the session prune. bquest
/// already carries Rust and already reads these shards at pack time,
/// so folding it in removes the replay leg's only external
/// dependency.
///
/// DECOMPRESSION IS STREAMED, and that is a storage requirement
/// rather than a refinement. The caller's loop fetches one shard,
/// rips it, and deletes it, which bounds peak storage at one
/// compressed file; materializing the decompressed shard first
/// multiplies that several-fold for no gain. So the reader decodes
/// line by line and stops reading the moment the budget is crossed.
///
/// IDENTITY RIDES METADATA, never the text. That is the lineage's own
/// presentation - ai2's renderers put the repository, path and
/// license in a sidecar and let the text field carry raw file bytes -
/// and departing from it would teach a header convention their model
/// never saw.
///
/// `source` carries the STREAM name rather than the upstream's own
/// source field, because a wayside audit counts documents per source
/// stream and that count is the evidence the excluded topics
/// contributed nothing.
pub(crate) fn mix_rip(args: &MixRipArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let file = fs::File::open(&args.shard)?;
    let decoder = match zstd::stream::read::Decoder::new(file) {
        Ok(decoder) => decoder,
        Err(e) => snafu::whatever!("{}: zstd decode failed: {e}", args.shard.display()),
    };
    let document_type = lib::nu::parse_typedef(MIX_DOCUMENT_TYPEDEF)?;
    let shard_path = match &args.shard_path {
        Some(path) => path.clone(),
        None => args
            .shard
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    let stamp = args.stamp.clone().unwrap_or_default();

    let mut documents: Vec<lib::nu::Value> = Vec::new();
    let mut text_bytes = 0usize;
    let mut seen = 0usize;
    let mut without_text = 0usize;
    for line in io::BufReader::new(decoder).lines() {
        // The budget gates the READ, so a met budget stops
        // decompressing rather than merely stops collecting.
        if text_bytes >= args.budget_bytes {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        let row: serde_json::Value = match serde_json::from_str(&line) {
            Ok(row) => row,
            Err(e) => {
                snafu::whatever!("{}:{seen}: JSON parse failed: {e}", args.shard.display())
            }
        };
        let text = match row.get("text").and_then(|text| text.as_str()) {
            Some(text) if !text.trim().is_empty() => text,
            _ => {
                without_text += 1;
                continue;
            }
        };
        let id = match row.get("id").and_then(|id| id.as_str()) {
            Some(id) => id.to_string(),
            None => format!("{}/{shard_path}#{seen}", args.name),
        };
        let document = lib::nu::Value::record(
            lib::nu::record! {
                "id" => v_str(&id),
                "text" => v_str(text),
                "source" => v_str(&args.name),
                "added" => v_str(&stamp),
                "created" => v_str(&stamp),
                "metadata" => lib::nu::Value::record(
                    lib::nu::record! {
                        "repo" => v_str(&args.hub),
                        "path" => v_str(&shard_path),
                        "language" => v_str("text"),
                        "license" => v_str(&args.license),
                        "kind" => v_str(&args.kind),
                    },
                    span(),
                ),
            },
            span(),
        );
        lib::nu::conform(&document, &document_type)?;
        text_bytes += text.len();
        documents.push(document);
    }

    let taken = documents.len();
    snafu::ensure_whatever!(
        taken > 0,
        "{}: no documents carried text (read {seen} rows)",
        args.shard.display()
    );
    lib::nu::save_value(&args.out, &lib::nu::Value::list(documents, span()))?;

    let provenance = lib::nu::Value::record(
        lib::nu::record! {
            "shard" => v_str(&args.shard.display().to_string()),
            "shard_path" => v_str(&shard_path),
            "name" => v_str(&args.name),
            "hub" => v_str(&args.hub),
            "kind" => v_str(&args.kind),
            "budget_bytes" => v_int(args.budget_bytes as i64),
            "documents_read" => v_int(seen as i64),
            "documents_taken" => v_int(taken as i64),
            "documents_without_text" => v_int(without_text as i64),
            "text_bytes" => v_int(text_bytes as i64),
            "bquest_version" => v_str(env!("CARGO_PKG_VERSION")),
        },
        span(),
    );
    lib::nu::save_value(&args.out.with_extension("provenance.nuon"), &provenance)?;

    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "name" => v_str(&args.name),
            "documents_taken" => v_int(taken as i64),
            "documents_read" => v_int(seen as i64),
            "text_bytes" => v_int(text_bytes as i64),
            "met" => v_bool(text_bytes >= args.budget_bytes),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
