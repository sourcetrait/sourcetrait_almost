//! The ImagineQuestAssociations store: the typed NUON association
//! artifact the matrix build and stage-zero corpus read.
use crate::*;

use std::io::BufRead;
use std::io::Write;

use crate::bucket::BucketTable;
use crate::dictionary::single_piece_folded;
use crate::lexer::BEGIN_REPEAT_ALIAS;
use crate::lexer::END_REPEAT_ALIAS;
use crate::lexer::KEYWORD_BEGIN_REPEAT;
use crate::lexer::KEYWORD_END_REPEAT;
use crate::lexer::KEYWORD_REPEAT;
use crate::lexer::KEYWORD_REPETITION;
use crate::lexer::REPEAT_ALIAS;
use crate::lexer::REPETITION_ALIAS;
use crate::ucd::CharacterTable;

/// A purely conceptual association target: an embedding-space anchor
/// with a keyword-page address and no wire legality.
///
/// Each concept is backed by an AbstractConceptMarker - the keyword
/// id that gives the concept its address; the marker is illegal in
/// wire, never emitted and never parsed. Repetition is the first:
/// every keyboard row and the three repetition operators associate
/// to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbstractConcept {
    Repetition,
}

impl AbstractConcept {
    /// Every concept, store order.
    pub(crate) const ALL: [Self; 1] = [Self::Repetition];

    /// The concept's name: the store's join key.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Repetition => "repetition",
        }
    }

    /// The AbstractConceptMarker: the keyword id backing the concept.
    pub(crate) fn marker(self) -> u32 {
        match self {
            Self::Repetition => KEYWORD_REPETITION,
        }
    }

    /// The marker's `<|...|>` spelling on the test surface.
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Repetition => REPETITION_ALIAS,
        }
    }
}

/// One gloss as one NUON-lines-safe line: whitespace runs collapse
/// to single spaces, ends trimmed.
pub(crate) fn sanitize_gloss(gloss: &str) -> String {
    gloss.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// The fields read off one wiktextract entry for the store; the rest
/// is skipped.
#[derive(serde::Deserialize)]
pub(crate) struct Entry {
    #[serde(default)]
    word: Option<String>,
    #[serde(default)]
    lang_code: Option<String>,
    #[serde(default)]
    pos: Option<String>,
    #[serde(default)]
    forms: Vec<Form>,
    #[serde(default)]
    senses: Vec<Sense>,
}

#[derive(serde::Deserialize)]
struct Form {
    #[serde(default)]
    form: Option<String>,
}

#[derive(serde::Deserialize)]
struct Sense {
    #[serde(default)]
    glosses: Vec<String>,
    #[serde(default)]
    form_of: Vec<FormOf>,
}

#[derive(serde::Deserialize)]
struct FormOf {
    #[serde(default)]
    word: Option<String>,
}

/// What the dump pass counted.
#[derive(Default)]
pub(crate) struct AssociationsTally {
    pub(crate) entries_read: usize,
    pub(crate) non_english: usize,
    pub(crate) senses_kept: usize,
    pub(crate) links_from_forms: usize,
    pub(crate) links_from_form_of: usize,
}

/// The dump-side accumulation: spellings, definitions, inflections.
pub(crate) struct DumpSide<'a> {
    table: &'a CharacterTable,
    /// Every folded single-piece word: headwords and forms, the
    /// dictionary pass's exact set.
    pub(crate) words: HashSet<String>,
    /// (word, pos) to its accumulated sense glosses, dump order.
    pub(crate) definitions: HashMap<(String, String), Vec<String>>,
    /// (form, lemma) links, both folded single-piece, form != lemma.
    pub(crate) links: HashSet<(String, String)>,
    pub(crate) tally: AssociationsTally,
}

impl<'a> DumpSide<'a> {
    pub(crate) fn new(table: &'a CharacterTable) -> Self {
        Self {
            table,
            words: HashSet::new(),
            definitions: HashMap::new(),
            links: HashSet::new(),
            tally: AssociationsTally::default(),
        }
    }

    /// Absorb one entry: headword and forms into the word set, a form
    /// linking to its headword lemma; each sense's most specific
    /// gloss (the last: wiktextract glosses refine parent-to-child)
    /// as a definition, and a form_of sense linking this entry to
    /// its lemma.
    pub(crate) fn absorb(&mut self, entry: &Entry) {
        self.tally.entries_read += 1;
        if entry.lang_code.as_deref() != Some("en") {
            self.tally.non_english += 1;
            return;
        }
        let headword = entry
            .word
            .as_deref()
            .and_then(|word| single_piece_folded(self.table, word));
        if let Some(word) = &headword {
            self.words.insert(word.clone());
        }
        for form in &entry.forms {
            let Some(form) = &form.form else { continue };
            let Some(folded) = single_piece_folded(self.table, form) else {
                continue;
            };
            self.words.insert(folded.clone());
            if let Some(lemma) = &headword
                && folded != *lemma
                && self.links.insert((folded, lemma.clone()))
            {
                self.tally.links_from_forms += 1;
            }
        }
        let Some(word) = headword else { return };
        let mut senses: Vec<String> = Vec::new();
        for sense in &entry.senses {
            if let Some(gloss) = sense.glosses.last() {
                let clean = sanitize_gloss(gloss);
                if !clean.is_empty() {
                    senses.push(clean);
                }
            }
            for form_of in &sense.form_of {
                let Some(lemma) = &form_of.word else { continue };
                let Some(lemma) = single_piece_folded(self.table, lemma) else {
                    continue;
                };
                if lemma != word && self.links.insert((word.clone(), lemma)) {
                    self.tally.links_from_form_of += 1;
                }
            }
        }
        let Some(pos) = entry.pos.as_deref().filter(|pos| !pos.is_empty()) else {
            return;
        };
        if senses.is_empty() {
            return;
        }
        self.tally.senses_kept += senses.len();
        self.definitions
            .entry((word, pos.to_string()))
            .or_default()
            .extend(senses);
    }
}

/// Stream the dump through one DumpSide.
fn extract_dump<'a>(table: &'a CharacterTable, dump: &Path) -> BiquestResult<DumpSide<'a>> {
    let file = fs::File::open(dump)?;
    let reader = io::BufReader::with_capacity(1 << 20, file);
    let mut side = DumpSide::new(table);
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: Entry = match serde_json::from_str(&line) {
            Ok(entry) => entry,
            Err(e) => {
                snafu::whatever!("{}:{}: JSON parse failed: {e}", dump.display(), index + 1)
            }
        };
        side.absorb(&entry);
    }
    snafu::ensure_whatever!(
        !side.words.is_empty(),
        "{}: no admissible words found",
        dump.display()
    );
    Ok(side)
}

/// One value as one condensed NUON line; a value that renders wide
/// (an unsanitized embedded newline) is refused rather than written.
pub(crate) fn condensed_line(
    engine_state: &nu_protocol::engine::EngineState,
    value: &harness::nu::Value,
) -> BiquestResult<String> {
    let config = nuon::ToNuonConfig::default().style(nuon::ToStyle::Raw);
    let text = match nuon::to_nuon(engine_state, value, config) {
        Ok(text) => text,
        Err(e) => snafu::whatever!("nuon render failed: {e}"),
    };
    snafu::ensure_whatever!(
        !text.contains('\n'),
        "store row rendered multi-line; sanitize the source value"
    );
    Ok(text)
}

/// Stream a table to a NUON-lines file, one condensed record per
/// line; returns the row count.
fn write_lines<I>(
    engine_state: &nu_protocol::engine::EngineState,
    path: &Path,
    rows: I,
) -> BiquestResult<usize>
where
    I: Iterator<Item = harness::nu::Value>,
{
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::with_capacity(1 << 20, file);
    let mut count = 0usize;
    for row in rows {
        let line = condensed_line(engine_state, &row)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
        count += 1;
    }
    writer.flush()?;
    Ok(count)
}

fn code_points_value(points: impl Iterator<Item = u32>) -> harness::nu::Value {
    harness::nu::Value::list(points.map(|point| v_int(point as i64)).collect(), span())
}

/// The abstract-concepts table: concept, marker, spelling.
fn abstract_concepts_value() -> harness::nu::Value {
    harness::nu::Value::list(
        AbstractConcept::ALL
            .iter()
            .map(|concept| {
                harness::nu::Value::record(
                    harness::nu::record! {
                        "concept" => v_str(concept.name()),
                        "marker" => v_int(concept.marker() as i64),
                        "spelling" => v_str(concept.spelling()),
                    },
                    span(),
                )
            })
            .collect(),
        span(),
    )
}

/// The repetition machinery's associations: every keyboard row to its
/// character, the repetition AbstractConcept, and its count; the
/// three operators to the concept. The wire no longer forces the
/// single convention - association hints it.
fn repetition_value(buckets: &BucketTable) -> harness::nu::Value {
    let concept = AbstractConcept::Repetition;
    let keyboard_rows: Vec<harness::nu::Value> = (0..buckets.count())
        .map(|index| {
            let entry = buckets.entry(index);
            harness::nu::Value::record(
                harness::nu::record! {
                    "row_index" => v_int(index as i64),
                    "code_point" => v_int(entry.symbol as u32 as i64),
                    "symbol" => v_str(&entry.symbol.to_string()),
                    "count" => v_int(entry.count as i64),
                    "concept" => v_str(concept.name()),
                },
                span(),
            )
        })
        .collect();
    let operators: Vec<harness::nu::Value> = [
        (KEYWORD_BEGIN_REPEAT, BEGIN_REPEAT_ALIAS, "begin_repeat"),
        (KEYWORD_END_REPEAT, END_REPEAT_ALIAS, "end_repeat"),
        (KEYWORD_REPEAT, REPEAT_ALIAS, "repeat"),
    ]
    .iter()
    .map(|&(keyword, spelling, operator)| {
        harness::nu::Value::record(
            harness::nu::record! {
                "operator" => v_str(operator),
                "keyword" => v_int(keyword as i64),
                "spelling" => v_str(spelling),
                "concept" => v_str(concept.name()),
            },
            span(),
        )
    })
    .collect();
    harness::nu::Value::record(
        harness::nu::record! {
            "keyboard_rows" => harness::nu::Value::list(keyboard_rows, span()),
            "operators" => harness::nu::Value::list(operators, span()),
        },
        span(),
    )
}

/// The simple case maps, each direction its own table, ascending.
fn case_pairs_value(table: &CharacterTable) -> harness::nu::Value {
    let mut uppercase: Vec<harness::nu::Value> = Vec::new();
    let mut lowercase: Vec<harness::nu::Value> = Vec::new();
    for row in &table.rows {
        if let Some(&target) = table.simple_uppercase.get(&row.code_point) {
            uppercase.push(harness::nu::Value::record(
                harness::nu::record! {
                    "code_point" => v_int(row.code_point as i64),
                    "uppercase" => v_int(target as i64),
                },
                span(),
            ));
        }
        if let Some(&target) = table.simple_lowercase.get(&row.code_point) {
            lowercase.push(harness::nu::Value::record(
                harness::nu::record! {
                    "code_point" => v_int(row.code_point as i64),
                    "lowercase" => v_int(target as i64),
                },
                span(),
            ));
        }
    }
    harness::nu::Value::record(
        harness::nu::record! {
            "uppercase" => harness::nu::Value::list(uppercase, span()),
            "lowercase" => harness::nu::Value::list(lowercase, span()),
        },
        span(),
    )
}

/// Canonical decompositions, ascending; Hangul stays algorithmic.
fn decompositions_value(table: &CharacterTable) -> harness::nu::Value {
    let rows = table
        .rows
        .iter()
        .filter_map(|row| {
            let points = table.decompositions.get(&row.code_point)?;
            Some(harness::nu::Value::record(
                harness::nu::record! {
                    "code_point" => v_int(row.code_point as i64),
                    "decomposition" => code_points_value(points.iter().copied()),
                },
                span(),
            ))
        })
        .collect();
    harness::nu::Value::list(rows, span())
}

/// Numeric values, ascending, verbatim strings (fractions included).
fn numeric_values_value(table: &CharacterTable) -> harness::nu::Value {
    let rows = table
        .rows
        .iter()
        .filter_map(|row| {
            let value = table.numeric_values.get(&row.code_point)?;
            Some(harness::nu::Value::record(
                harness::nu::record! {
                    "code_point" => v_int(row.code_point as i64),
                    "value" => v_str(value),
                },
                span(),
            ))
        })
        .collect();
    harness::nu::Value::list(rows, span())
}

/// Family membership: one row per family name, members ascending.
fn family_rows(
    names: &[String],
    key: &str,
    member_of: impl Fn(&crate::ucd::CharRow) -> Option<u16>,
    table: &CharacterTable,
) -> harness::nu::Value {
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); names.len()];
    for row in &table.rows {
        if let Some(index) = member_of(row) {
            members[index as usize].push(row.code_point);
        }
    }
    let mut named: Vec<(&String, Vec<u32>)> = names.iter().zip(members).collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    harness::nu::Value::list(
        named
            .into_iter()
            .map(|(name, points)| {
                harness::nu::Value::record(
                    harness::nu::record! {
                        key => v_str(name),
                        "code_points" => code_points_value(points.into_iter()),
                    },
                    span(),
                )
            })
            .collect(),
        span(),
    )
}

/// `biquest associations build`: the whole store from the embedded
/// UCD and a wiktextract dump, one class per file under --out.
pub(crate) fn associations_build(args: &AssociationsBuildArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let table = CharacterTable::embedded()?;
    let buckets = BucketTable::new();
    let engine_state = nu_protocol::engine::EngineState::new();
    fs::create_dir_all(&args.out)?;

    harness::nu::save_value(
        &args.out.join("abstract_concepts.nuon"),
        &abstract_concepts_value(),
    )?;
    harness::nu::save_value(&args.out.join("repetition.nuon"), &repetition_value(&buckets))?;
    harness::nu::save_value(&args.out.join("case_pairs.nuon"), &case_pairs_value(&table))?;
    harness::nu::save_value(
        &args.out.join("decompositions.nuon"),
        &decompositions_value(&table),
    )?;
    harness::nu::save_value(
        &args.out.join("numeric_values.nuon"),
        &numeric_values_value(&table),
    )?;
    harness::nu::save_value(
        &args.out.join("scripts.nuon"),
        &family_rows(
            &table.scripts,
            "script",
            |row| (row.script != u16::MAX).then_some(row.script),
            &table,
        ),
    )?;
    harness::nu::save_value(
        &args.out.join("categories.nuon"),
        &family_rows(
            &table.general_categories,
            "category",
            |row| Some(row.category as u16),
            &table,
        ),
    )?;

    let side = extract_dump(&table, &args.dump)?;

    let mut words: Vec<&String> = side.words.iter().collect();
    words.sort();
    let spelling_rows = write_lines(
        &engine_state,
        &args.out.join("spellings.nuonl"),
        words.iter().map(|word| {
            harness::nu::Value::record(
                harness::nu::record! {
                    "word" => v_str(word),
                    "code_points" => code_points_value(word.chars().map(|c| c as u32)),
                },
                span(),
            )
        }),
    )?;

    let mut definition_keys: Vec<&(String, String)> = side.definitions.keys().collect();
    definition_keys.sort();
    let definition_rows = write_lines(
        &engine_state,
        &args.out.join("definitions.nuonl"),
        definition_keys.iter().map(|key| {
            let senses = &side.definitions[*key];
            harness::nu::Value::record(
                harness::nu::record! {
                    "word" => v_str(&key.0),
                    "pos" => v_str(&key.1),
                    "senses" => harness::nu::Value::list(
                        senses.iter().map(|sense| v_str(sense)).collect(),
                        span(),
                    ),
                },
                span(),
            )
        }),
    )?;

    // A form_of lemma can name an entry the dump never headwords;
    // the written links stay closed over the spelling rows.
    let mut links: Vec<&(String, String)> = side
        .links
        .iter()
        .filter(|(form, lemma)| side.words.contains(form) && side.words.contains(lemma))
        .collect();
    links.sort();
    let links_outside_words = side.links.len() - links.len();
    let inflection_rows = write_lines(
        &engine_state,
        &args.out.join("inflections.nuonl"),
        links.iter().map(|(form, lemma)| {
            harness::nu::Value::record(
                harness::nu::record! {
                    "form" => v_str(form),
                    "lemma" => v_str(lemma),
                },
                span(),
            )
        }),
    )?;

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "dump" => v_str(&args.dump.display().to_string()),
            "unicode_assigned" => v_int(table.assigned_count() as i64),
            "entries_read" => v_int(side.tally.entries_read as i64),
            "non_english" => v_int(side.tally.non_english as i64),
            "spelling_rows" => v_int(spelling_rows as i64),
            "definition_rows" => v_int(definition_rows as i64),
            "senses_kept" => v_int(side.tally.senses_kept as i64),
            "inflection_rows" => v_int(inflection_rows as i64),
            "links_from_forms" => v_int(side.tally.links_from_forms as i64),
            "links_from_form_of" => v_int(side.tally.links_from_form_of as i64),
            "links_outside_words" => v_int(links_outside_words as i64),
            "uppercase_maps" => v_int(table.simple_uppercase.len() as i64),
            "lowercase_maps" => v_int(table.simple_lowercase.len() as i64),
            "decomposition_rows" => v_int(table.decompositions.len() as i64),
            "numeric_value_rows" => v_int(table.numeric_values.len() as i64),
            "script_families" => v_int(table.scripts.len() as i64),
            "category_families" => v_int(table.general_categories.len() as i64),
            "keyboard_rows" => v_int(buckets.count() as i64),
            "operator_rows" => v_int(3),
            "abstract_concepts" => v_int(AbstractConcept::ALL.len() as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.join("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "entries_read" => v_int(side.tally.entries_read as i64),
            "spelling_rows" => v_int(spelling_rows as i64),
            "definition_rows" => v_int(definition_rows as i64),
            "inflection_rows" => v_int(inflection_rows as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
