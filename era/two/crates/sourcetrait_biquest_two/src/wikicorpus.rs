//! The stage-zero corpus emitter: the ruled first attempt - depth 1,
//! pass 1, dictionary side only - every vocabulary word's own
//! document to a sharded markdown tree.
use crate::*;

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::io::Write;

use crate::dictionary::read_words_ordered;
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

/// `biquest wikimedia corpus`: emit the dictionary-side documents
/// for the whole vocabulary (or --words), sharded under --out, with
/// an aggregate audit, the missing-word list, and provenance.
pub(crate) fn wikimedia_corpus(args: &WikimediaCorpusArgs) -> BiquestResult<()> {
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
