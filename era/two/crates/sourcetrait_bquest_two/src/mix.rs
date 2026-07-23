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
