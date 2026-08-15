//! The provenance derivation: the word set, definitions, and
//! inflection links from the raw enwiktionary dump, through the same
//! renderer the corpus uses - the wiktextract replacement.
use crate::*;

use crate::lexer::whole_candidate_folded;
use crate::ucd::CharacterTable;
use crate::wikidoc::derive_page;
use crate::wikixml::open_pages;
use crate::wikixml::PageReader;
use crate::wikixml::WikiPage;

/// What one dump pass counted.
#[derive(Default)]
pub(crate) struct DeriveTally {
    pub(crate) pages_read: usize,
    pub(crate) ns0_pages: usize,
    pub(crate) redirect_pages: usize,
    pub(crate) english_pages: usize,
    pub(crate) no_entry_pages: usize,
    pub(crate) parse_failures: usize,
    pub(crate) titles_admitted: usize,
    pub(crate) forms_seen: usize,
    pub(crate) forms_admitted: usize,
    pub(crate) senses_kept: usize,
    pub(crate) links_from_forms: usize,
    pub(crate) links_from_form_of: usize,
}

/// The dump-side derivation: every folded admissible word (titles
/// plus head-derived forms), the per-(word, pos) rendered senses,
/// and the folded (form, lemma) links.
pub(crate) struct DumpDerivation {
    pub(crate) words: HashSet<String>,
    /// (folded word, lowercased pos section) to its senses, page
    /// order within a word.
    pub(crate) definitions: HashMap<(String, String), Vec<String>>,
    /// (form, lemma) links, both folded, form != lemma; NOT yet
    /// closed over the word set - the writer filters.
    pub(crate) links: HashSet<(String, String)>,
    pub(crate) tally: DeriveTally,
}

/// Whether a page's wikitext carries an English h2 line - the cheap
/// pre-filter ahead of the full parse (the parse confirms).
fn mentions_english_heading(text: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("==")
            && !trimmed.starts_with("===")
            && trimmed.ends_with("==")
            && trimmed.trim_matches('=').trim() == "English"
    })
}

impl DumpDerivation {
    fn new() -> Self {
        Self {
            words: HashSet::new(),
            definitions: HashMap::new(),
            links: HashSet::new(),
            tally: DeriveTally::default(),
        }
    }

    /// Absorb one English-bearing ns0 page: the folded title and the
    /// admissible derived forms into the word set, form-to-title
    /// links, per-POS senses under the folded title, and form-of
    /// lemma links. A page the parser faults counts and skips - the
    /// derivation never dies on one page.
    fn absorb(&mut self, table: &CharacterTable, page: &WikiPage) {
        let derived = match derive_page(page) {
            Ok(derived) => derived,
            Err(_) => {
                self.tally.parse_failures += 1;
                return;
            }
        };
        if !derived.has_english {
            return;
        }
        self.tally.english_pages += 1;
        if derived.no_entry {
            // The wiki's own declaration: no English entry exists for
            // this spelling, so the title is not a vocabulary word.
            self.tally.no_entry_pages += 1;
            return;
        }
        let title = whole_candidate_folded(table, &page.title);
        if let Some(word) = &title {
            self.tally.titles_admitted += 1;
            self.words.insert(word.clone());
        }
        for form in &derived.forms {
            self.tally.forms_seen += 1;
            let Some(folded) = whole_candidate_folded(table, form) else {
                continue;
            };
            self.tally.forms_admitted += 1;
            self.words.insert(folded.clone());
            if let Some(word) = &title
                && folded != *word
                && self.links.insert((folded, word.clone()))
            {
                self.tally.links_from_forms += 1;
            }
        }
        let Some(word) = title else { return };
        for (pos, sense) in &derived.senses {
            let clean = crate::associations::sanitize_gloss(sense);
            if clean.is_empty() {
                continue;
            }
            self.tally.senses_kept += 1;
            self.definitions
                .entry((word.clone(), pos.to_lowercase()))
                .or_default()
                .push(clean);
        }
        for lemma in &derived.form_of_lemmas {
            let Some(lemma) = whole_candidate_folded(table, lemma) else {
                continue;
            };
            if lemma != word && self.links.insert((word.clone(), lemma)) {
                self.tally.links_from_form_of += 1;
            }
        }
    }
}

/// One pass over a page stream: ns0, non-redirect, English-bearing
/// pages absorb; everything else only counts.
pub(crate) fn derive_pages<R: io::BufRead>(
    table: &CharacterTable,
    reader: &mut PageReader<R>,
    progress: bool,
) -> BiquestResult<DumpDerivation> {
    let started = std::time::Instant::now();
    let mut derivation = DumpDerivation::new();
    while let Some(page) = reader.next_page()? {
        derivation.tally.pages_read += 1;
        if progress && derivation.tally.pages_read.is_multiple_of(1_000_000) {
            eprintln!(
                "derive: {} pages, {} english, {:.0} s",
                derivation.tally.pages_read,
                derivation.tally.english_pages,
                started.elapsed().as_secs_f64()
            );
        }
        if page.ns != 0 {
            continue;
        }
        derivation.tally.ns0_pages += 1;
        if page.redirect.is_some() {
            derivation.tally.redirect_pages += 1;
            continue;
        }
        if !mentions_english_heading(&page.text) {
            continue;
        }
        derivation.absorb(table, &page);
    }
    snafu::ensure_whatever!(
        !derivation.words.is_empty(),
        "the page stream yields no admissible words"
    );
    Ok(derivation)
}

/// The whole-dump pass: an export .xml or a pages-articles dump
/// (.xml.bz2), streamed across every bz2 stream.
pub(crate) fn derive_dump(
    table: &CharacterTable,
    source: &Path,
    progress: bool,
) -> BiquestResult<DumpDerivation> {
    let mut reader = open_pages(source)?;
    derive_pages(table, &mut reader, progress)
}
