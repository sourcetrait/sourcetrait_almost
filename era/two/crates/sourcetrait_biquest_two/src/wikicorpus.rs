//! The stage-zero corpus emitter: the ruled first attempt - depth 1,
//! pass 1, dictionary side only - every vocabulary word's own
//! document to a sharded markdown tree.
use crate::*;

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::io::Write;

use crate::assembler::Assembler;
use crate::assembler::SyntaxTable;
use crate::bucket::BucketTable;
use crate::dictionary::read_words_ordered;
use crate::lexer::Segmenter;
use crate::ucd::CharacterTable;
use crate::wikidoc::order_word_pages;
use crate::wikidoc::render_word_document;
use crate::wikixml::index_matches;
use crate::wikixml::read_block;
use crate::wikixml::IndexRow;
use crate::wikixml::WikiPage;

/// A word's two-character shard directory: ascii alphanumerics
/// lowercased, anything else an underscore.
pub(crate) fn shard_of(word: &str) -> String {
    word.chars()
        .take(2)
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// A small LRU over decoded multistream blocks: words iterate in
/// offset order, so runs share a block, and a word whose pages span
/// far-apart blocks still finds the earlier one cached.
struct BlockCache {
    blocks: HashMap<u64, Vec<WikiPage>>,
    order: VecDeque<u64>,
    capacity: usize,
}

impl BlockCache {
    fn new(capacity: usize) -> Self {
        Self { blocks: HashMap::new(), order: VecDeque::new(), capacity }
    }

    fn pages(
        &mut self,
        source: &Path,
        offset: u64,
        reads: &mut usize,
    ) -> BiquestResult<&Vec<WikiPage>> {
        if !self.blocks.contains_key(&offset) {
            let block = read_block(source, offset)?;
            *reads += 1;
            if self.order.len() >= self.capacity
                && let Some(evicted) = self.order.pop_front()
            {
                self.blocks.remove(&evicted);
            }
            self.order.push_back(offset);
            self.blocks.insert(offset, block);
        }
        Ok(&self.blocks[&offset])
    }
}

/// `biquest wiktionary corpus`: emit the dictionary-side documents
/// for the whole vocabulary (or --words), sharded under --out, with
/// an aggregate audit, the missing-word list, and provenance.
pub(crate) fn wiktionary_corpus(args: &WiktionaryCorpusArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let words = match &args.words {
        Some(path) => read_words_ordered(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let vocabulary: HashSet<&str> = words.iter().map(String::as_str).collect();

    // One index scan builds the folded word-to-rows map in memory;
    // colon-carrying titles (namespaced pages) never match.
    let mut rows_of: HashMap<String, Vec<IndexRow>> = HashMap::new();
    {
        let mut matcher = |title: &str| -> bool {
            if title.contains(':') {
                return false;
            }
            let Ok(folded) = table.fold_str(title) else { return false };
            vocabulary.contains(folded.as_str())
        };
        for row in index_matches(&args.index, &mut matcher)? {
            let folded = match table.fold_str(&row.title) {
                Ok(folded) => folded,
                Err(_) => continue,
            };
            rows_of.entry(folded).or_default().push(row);
        }
    }

    // Words ordered by their first block offset, so the dump reads
    // nearly sequentially through the cache.
    let mut by_offset: BTreeMap<(u64, String), &str> = BTreeMap::new();
    for word in &words {
        if let Some(rows) = rows_of.get(word.as_str()) {
            let first = rows.iter().map(|row| row.offset).min().unwrap_or(0);
            by_offset.insert((first, word.clone()), word.as_str());
        }
    }

    fs::create_dir_all(&args.out)?;
    let audit_path = args.out.join("audit.nuonl");
    let mut audit_file = io::BufWriter::new(fs::File::create(&audit_path)?);
    let engine_state = nu_protocol::engine::EngineState::new();

    let mut cache = BlockCache::new(16);
    let mut blocks_read = 0usize;
    let mut words_rendered = 0usize;
    let mut words_empty = 0usize;
    let mut pages_rendered = 0usize;
    let mut render_failures = 0usize;
    let mut audit_rows = 0usize;
    for word in by_offset.values() {
        if let Some(limit) = args.limit
            && words_rendered >= limit
        {
            break;
        }
        let rows = &rows_of[*word];
        let mut pages: Vec<WikiPage> = Vec::with_capacity(rows.len());
        for row in rows {
            let block = cache.pages(&args.source, row.offset, &mut blocks_read)?;
            pages.extend(
                block.iter().filter(|page| page.title == row.title).cloned(),
            );
        }
        order_word_pages(word, &mut pages);
        if pages.is_empty() {
            continue;
        }
        let document = match render_word_document(&pages) {
            Ok(document) => document,
            Err(error) => {
                render_failures += 1;
                let row = harness::nu::Value::record(
                    harness::nu::record! {
                        "page" => v_str(word),
                        "class" => v_str("render_failed"),
                        "detail" => v_str(&error.to_string()),
                    },
                    span(),
                );
                writeln!(
                    audit_file,
                    "{}",
                    crate::associations::condensed_line(&engine_state, &row)?
                )?;
                audit_rows += 1;
                continue;
            }
        };
        // A document with nothing beyond its h1 (a fold-matched page
        // set with no English content) writes as <word>.empty.md -
        // skippable by suffix, existence still known (TheUser).
        let has_content = document
            .markdown
            .lines()
            .skip(1)
            .any(|line| !line.trim().is_empty());
        let filename = if has_content {
            format!("{word}.md")
        } else {
            words_empty += 1;
            let row = harness::nu::Value::record(
                harness::nu::record! {
                    "word" => v_str(word),
                    "page" => v_str(word),
                    "class" => v_str("document_empty"),
                    "detail" => v_str("no english content"),
                },
                span(),
            );
            writeln!(
                audit_file,
                "{}",
                crate::associations::condensed_line(&engine_state, &row)?
            )?;
            audit_rows += 1;
            format!("{word}.empty.md")
        };
        let shard_dir = args.out.join(shard_of(word));
        fs::create_dir_all(&shard_dir)?;
        fs::write(shard_dir.join(filename), &document.markdown)?;
        words_rendered += 1;
        pages_rendered += pages.len();
        for row in &document.audit {
            let record = harness::nu::Value::record(
                harness::nu::record! {
                    "word" => v_str(word),
                    "page" => v_str(&row.page),
                    "class" => v_str(&row.class),
                    "detail" => v_str(&row.detail),
                },
                span(),
            );
            writeln!(
                audit_file,
                "{}",
                crate::associations::condensed_line(&engine_state, &record)?
            )?;
            audit_rows += 1;
        }
        if words_rendered.is_multiple_of(25_000) {
            eprintln!(
                "corpus: {words_rendered} words, {blocks_read} blocks, {:.0} s",
                started.elapsed().as_secs_f64()
            );
        }
    }
    audit_file.flush()?;

    // The empty tails, measured: vocabulary words with no page. The
    // list is the word-keyed Wikipedia stage's target input.
    let missing: Vec<&str> = words
        .iter()
        .map(String::as_str)
        .filter(|word| !rows_of.contains_key(*word))
        .collect();
    let mut missing_payload = String::with_capacity(missing.len() * 10);
    for word in &missing {
        missing_payload.push_str(word);
        missing_payload.push('\n');
    }
    fs::write(args.out.join("missing_words.txt"), missing_payload)?;

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "source" => v_str(&args.source.display().to_string()),
            "index" => v_str(&args.index.display().to_string()),
            "scope" => v_str("depth 1, pass 1, dictionary side only"),
            "vocabulary_words" => v_int(words.len() as i64),
            "words_with_pages" => v_int(rows_of.len() as i64),
            "words_rendered" => v_int(words_rendered as i64),
            "words_empty" => v_int(words_empty as i64),
            "words_missing" => v_int(missing.len() as i64),
            "pages_rendered" => v_int(pages_rendered as i64),
            "render_failures" => v_int(render_failures as i64),
            "audit_rows" => v_int(audit_rows as i64),
            "blocks_read" => v_int(blocks_read as i64),
            "limit" => match args.limit {
                Some(limit) => v_int(limit as i64),
                None => harness::nu::Value::nothing(span()),
            },
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.join("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "words_rendered" => v_int(words_rendered as i64),
            "words_empty" => v_int(words_empty as i64),
            "words_missing" => v_int(missing.len() as i64),
            "pages_rendered" => v_int(pages_rendered as i64),
            "render_failures" => v_int(render_failures as i64),
            "audit_rows" => v_int(audit_rows as i64),
            "blocks_read" => v_int(blocks_read as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}

/// The template's content placeholder, replaced with the fill.
const CONTENT_PLACEHOLDER: &str = "{{ content }}";
/// The template's page placeholder, replaced inline.
const PAGE_PLACEHOLDER: &str = "{{ page }}";
/// The widest page field the corpus can produce, so the measured
/// frame overhead is conservative for every real call.
const WIDE_PAGE_FIELD: &str = "1000000..1027171";

/// Render the page template as the quill build renders it: the page
/// field inline, the content fill riding the placeholder line's own
/// indent on every line after the first (empty lines empty) -
/// liquid's verbatim-insertion semantics replicated, so a measured
/// page and its built quill are the same bytes.
fn render_page_template(
    template: &str,
    page_field: &str,
    content: &str,
) -> BiquestResult<String> {
    let Some(placeholder_line) = template
        .lines()
        .find(|line| line.contains(CONTENT_PLACEHOLDER))
    else {
        snafu::whatever!("the page template carries no {CONTENT_PLACEHOLDER}");
    };
    snafu::ensure_whatever!(
        template.contains(PAGE_PLACEHOLDER),
        "the page template carries no {PAGE_PLACEHOLDER}"
    );
    let indent: String = placeholder_line
        .chars()
        .take_while(|c| *c == ' ')
        .collect();
    let mut fill = String::with_capacity(content.len() + content.len() / 8);
    for (index, line) in content.lines().enumerate() {
        if index > 0 {
            fill.push('\n');
            if !line.is_empty() {
                fill.push_str(&indent);
            }
        }
        fill.push_str(line);
    }
    Ok(template
        .replace(PAGE_PLACEHOLDER, page_field)
        .replace(CONTENT_PLACEHOLDER, &fill))
}

/// `biquest wiktionary railroad`: build the corpus into a railroad.
/// One page per word document lands under
/// corpus/derived/wikimedia/wiktionary (shards kept, empty documents
/// skipped); pages.nuonl carries each page's wire measure through the
/// rendered template frame, in page order; provenance carries the
/// probed frame overhead and per-join cost the quill build packs
/// with. The build commits as the railroad's next REV.
pub(crate) fn wiktionary_railroad(args: &WiktionaryRailroadArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let admitted = match &args.words {
        Some(path) => read_words_ordered(path)?,
        None => crate::dictionary::embedded_words(),
    };
    let segmenter = Segmenter::new(&table, &buckets, &admitted);
    let syntax = SyntaxTable::embedded()?;
    let assembler = Assembler::new(&syntax, &segmenter, &table, &buckets, &admitted);
    let template = fs::read_to_string(&args.page_template)?;

    let measure = |content: &str| -> BiquestResult<usize> {
        let framed = render_page_template(&template, WIDE_PAGE_FIELD, content)?;
        Ok(assembler.encode(&framed)?.len())
    };
    let overhead = measure("")?;
    // One more whole document in a call costs the blank separator
    // line plus the indent the fill puts on its first line - probed
    // through the real render rather than assumed.
    let single = (measure("# x\n")? - overhead) as i64;
    let joined = (measure("# x\n\n# x\n")? - overhead) as i64;
    let join_cost = joined - single - single;
    snafu::ensure_whatever!(
        join_cost >= 0,
        "the join probe measured negative ({join_cost}); the fill semantics drifted"
    );
    let join_cost = join_cost as usize;

    let railroad = match &args.railroad {
        Some(dir) => harness::railroad::Railroad::at(dir)?,
        None => harness::railroad::Railroad::lay()?,
    };
    let target = railroad.dir().join("corpus/derived/wikimedia/wiktionary");
    snafu::ensure_whatever!(
        !target.exists(),
        "{} is already occupied; build into a fresh railroad or clear it first",
        target.display()
    );
    fs::create_dir_all(&target)?;

    let mut shard_dirs: Vec<PathBuf> = fs::read_dir(&args.corpus)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    shard_dirs.sort();

    let engine_state = nu_protocol::engine::EngineState::new();
    let mut pages_file =
        io::BufWriter::new(fs::File::create(target.join("pages.nuonl"))?);
    let mut page = 0usize;
    let mut skipped_empty = 0usize;
    let mut wire_total = 0usize;
    let mut max_wire = 0usize;
    for shard_dir in &shard_dirs {
        let Some(shard) = shard_dir.file_name().and_then(|name| name.to_str())
        else {
            continue;
        };
        let mut files: Vec<PathBuf> = fs::read_dir(shard_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
            .collect();
        files.sort();
        let mut shard_made = false;
        for file in &files {
            let Some(name) = file.file_name().and_then(|name| name.to_str())
            else {
                continue;
            };
            if name.ends_with(".empty.md") {
                skipped_empty += 1;
                continue;
            }
            let Some(word) = name.strip_suffix(".md") else { continue };
            let text = fs::read_to_string(file)?;
            let wire = measure(&text)? - overhead;
            page += 1;
            wire_total += wire;
            max_wire = max_wire.max(wire);
            if !shard_made {
                fs::create_dir_all(target.join(shard))?;
                shard_made = true;
            }
            fs::copy(file, target.join(shard).join(name))?;
            let row = harness::nu::Value::record(
                harness::nu::record! {
                    "page" => v_int(page as i64),
                    "word" => v_str(word),
                    "shard" => v_str(shard),
                    "file" => v_str(&format!("{shard}/{name}")),
                    "wire_tokens" => v_int(wire as i64),
                },
                span(),
            );
            writeln!(
                pages_file,
                "{}",
                crate::associations::condensed_line(&engine_state, &row)?
            )?;
            if page.is_multiple_of(100_000) {
                eprintln!(
                    "railroad: {page} pages, {:.0} s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    pages_file.flush()?;
    snafu::ensure_whatever!(page > 0, "the corpus yields no pages");

    let corpus_provenance = harness::nu::from_nuon_text(&fs::read_to_string(
        args.corpus.join("provenance.nuon"),
    )?)?;
    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "source" => v_str("Wiktionary"),
            "pages" => v_int(page as i64),
            "skipped_empty" => v_int(skipped_empty as i64),
            "frame_overhead_wire" => v_int(overhead as i64),
            "join_cost_wire" => v_int(join_cost as i64),
            "page_field_probe" => v_str(WIDE_PAGE_FIELD),
            "mean_page_wire" => v_int((wire_total / page) as i64),
            "max_page_wire" => v_int(max_wire as i64),
            "page_template" => v_str(&args.page_template.display().to_string()),
            "corpus_dir" => v_str(&args.corpus.display().to_string()),
            "corpus" => corpus_provenance,
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&target.join("provenance.nuon"), &provenance)?;
    let rev = railroad.commit()?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "railroad" => v_str(&railroad.dir().display().to_string()),
            "nom" => v_str(railroad.nom().as_str()),
            "rev" => v_int(rev as i64),
            "pages" => v_int(page as i64),
            "skipped_empty" => v_int(skipped_empty as i64),
            "mean_page_wire" => v_int((wire_total / page) as i64),
            "max_page_wire" => v_int(max_wire as i64),
            "frame_overhead_wire" => v_int(overhead as i64),
            "join_cost_wire" => v_int(join_cost as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
