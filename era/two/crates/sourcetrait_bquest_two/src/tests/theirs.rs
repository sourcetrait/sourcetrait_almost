//! Theirs-conversion units: md wrapping, MarkerEscape, the refusal
//! split, the quest-posture system strip, and the budget walk - all
//! against the real tokenizer, which is the checkpoint-backed posture.
use crate::*;
use crate::theirs::{
    POOL_ROWS_TYPEDEF,
    TheirsOptions,
    convert_rows,
    read_pool_rows,
};

use std::sync::OnceLock;

fn tokenizer() -> &'static tokenizers::Tokenizer {
    static TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let dir = llm::model_dir(llm::consts::DPO_MODEL_NAME).expect("model dir");
        llm::load_tokenizer(&dir).expect("the checkpoint tokenizer loads")
    })
}

fn temp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("bquest_theirs_{tag}_{}", std::process::id()))
}

fn options(budget_tokens: usize) -> TheirsOptions {
    TheirsOptions {
        budget_tokens,
        seq_len: 2048,
        seed: 1,
        english: false,
        no_code_fences: false,
    }
}

fn pool_value(rows: &[(&str, &[(&str, &str)])]) -> harness::nu::Value {
    let values: Vec<harness::nu::Value> = rows
        .iter()
        .map(|(id, messages)| {
            let message_values: Vec<harness::nu::Value> = messages
                .iter()
                .map(|(role, content)| {
                    harness::nu::Value::record(
                        harness::nu::record! {
                            "role" => v_str(role),
                            "content" => v_str(content),
                        },
                        span(),
                    )
                })
                .collect();
            harness::nu::Value::record(
                harness::nu::record! {
                    "id" => v_str(id),
                    "source" => v_str("probe_class"),
                    "messages" => harness::nu::Value::list(message_values, span()),
                },
                span(),
            )
        })
        .collect();
    harness::nu::Value::list(values, span())
}

fn load_pool(tag: &str, rows: &[(&str, &[(&str, &str)])]) -> Vec<crate::theirs::PoolRow> {
    let root = temp_root(tag);
    std::fs::create_dir_all(&root).expect("temp root");
    let path = root.join("rows.nuon");
    harness::nu::save_value(&path, &pool_value(rows)).expect("rows write");
    let pool = read_pool_rows(&path).expect("rows read");
    let _ = std::fs::remove_dir_all(&root);
    pool
}

#[test]
fn conversion_wraps_finals_and_the_wrapped_turn_survives_the_codec() {
    let rows: [(&str, &[(&str, &str)]); 1] = [(
        "row_1",
        &[
            ("user", "Explain the water cycle in two short paragraphs."),
            (
                "assistant",
                "Water evaporates from oceans and lakes.\n\nIt condenses into \
                 clouds and returns as rain.",
            ),
        ],
    )];
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("wrap", &rows), &options(1)).expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert!(tally.supervised_tokens > 0);

    let table = harness::nu::Value::list(taken, span());
    harness::nu::conform(
        &table,
        &harness::nu::parse_typedef(example::SFT_TYPEDEF).expect("typedef"),
    )
    .expect("the part conforms to the supervised example shape");
    harness::nu::conform(
        &table,
        &harness::nu::parse_typedef(POOL_ROWS_TYPEDEF).expect("typedef"),
    )
    .expect("provenance rides in the rows");

    let row = table.as_list().expect("rows")[0].as_record().expect("record");
    let messages = field_rows(row, "messages").expect("messages");
    let user = field_str(messages[0], "content").expect("user content");
    let answer = field_str(messages[1], "content").expect("assistant content");
    assert_eq!(user, "Explain the water cycle in two short paragraphs.");
    assert!(
        answer.starts_with("<|output|> md\n") && answer.ends_with("\n<|/|>"),
        "the final wraps as an authored md block: {answer}"
    );

    // The codec gate the converter ran, re-proven here: the authored
    // turn translates and parses back as one md output block whose
    // content is the prose verbatim.
    let wire = harness::channel::authoring_to_wire(&answer);
    let blocks = harness::channel::parse_blocks(&wire, harness::channel::Aliasing::Strict)
        .expect("the wrapped turn parses on the wire");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].tag, harness::channel::Tag::Output);
    assert_eq!(blocks[0].header, "md");
    assert!(blocks[0].content.contains("evaporates"));
    assert!(blocks[0].content.contains("\n\n"), "structural newlines survive");
}

#[test]
fn marker_spellings_escape_and_the_tool_family_refuses() {
    let rows: [(&str, &[(&str, &str)]); 2] = [
        (
            "escaped",
            &[
                ("user", "What closes a block?"),
                ("assistant", "The closer is spelled <|extra_id_6|> on the wire."),
            ],
        ),
        (
            "refused",
            &[
                ("user", "What do tools ride?"),
                ("assistant", "Calls ride <function_calls> in that lineage."),
            ],
        ),
    ];
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("escape", &rows), &options(10_000))
            .expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert_eq!(tally.refused_markers, 1, "the non-pipe family refuses the row");
    assert_eq!(tally.escaped_rows, 1);

    let row = taken[0].as_record().expect("record");
    let answer = field_str(field_rows(row, "messages").expect("messages")[1], "content")
        .expect("content");
    assert!(
        answer.contains("<\\|extra_id_6|>"),
        "the marker introducer is broken: {answer}"
    );
    // The block's own opener and closer are real markers by design; the
    // guarantee is that the CONTENT between them carries no unbroken one.
    let wire = harness::channel::authoring_to_wire(&answer);
    let blocks = harness::channel::parse_blocks(&wire, harness::channel::Aliasing::Strict)
        .expect("parses");
    assert!(
        !harness::channel::carries_marker(&blocks[0].content),
        "no unbroken marker spelling survives in content: {}",
        blocks[0].content
    );
}

#[test]
fn the_refusal_split_names_its_causes() {
    let rows: [(&str, &[(&str, &str)]); 3] = [
        ("no_assistant", &[("user", "A question nothing answers.")]),
        ("foreign_role", &[("tool", "a tool turn"), ("assistant", "An answer.")]),
        ("empty", &[("user", "   "), ("assistant", "An answer.")]),
    ];
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("refuse", &rows), &options(10_000))
            .expect("converts");
    assert!(taken.is_empty());
    assert_eq!(tally.refused_no_assistant, 1);
    assert_eq!(tally.refused_role, 1);
    assert_eq!(tally.refused_empty, 1);
}

#[test]
fn a_foreign_system_turn_strips_to_the_quest_posture() {
    let rows: [(&str, &[(&str, &str)]); 1] = [(
        "sys",
        &[
            ("system", "You are a poetry bot."),
            ("user", "Say hello."),
            ("assistant", "Hello there."),
        ],
    )];
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("system", &rows), &options(1)).expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert_eq!(tally.stripped_system, 1);
    let row = taken[0].as_record().expect("record");
    let messages = field_rows(row, "messages").expect("messages");
    assert_eq!(messages.len(), 2, "the foreign system turn is gone");
    assert_eq!(field_str(messages[0], "role").expect("role"), "user");
}

#[test]
fn the_english_filter_refuses_a_row_that_does_not_read_english() {
    let rows: [(&str, &[(&str, &str)]); 2] = [
        (
            "spanish",
            &[
                ("user", "¿Cuál es la capital de Francia?"),
                ("assistant", "La capital de Francia es París, una ciudad europea."),
            ],
        ),
        (
            "english",
            &[
                ("user", "What is the capital of France?"),
                ("assistant", "The capital of France is Paris, and it is a large city."),
            ],
        ),
    ];
    let mut english_options = options(10_000);
    english_options.english = true;
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("english", &rows), &english_options)
            .expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert_eq!(tally.refused_language, 1);
    let row = taken[0].as_record().expect("record");
    assert_eq!(field_str(row, "id").expect("id"), "english");
}

#[test]
fn the_fence_filter_refuses_a_code_subject_row_only_when_asked() {
    let rows: [(&str, &[(&str, &str)]); 2] = [
        (
            "code",
            &[
                ("user", "Write a loop."),
                ("assistant", "Sure:\n\n```js\nfor (;;) {}\n```\n\nThat spins forever."),
            ],
        ),
        (
            "prose",
            &[
                ("user", "Why do leaves fall?"),
                ("assistant", "Trees shed leaves to conserve water in winter."),
            ],
        ),
    ];
    let mut fenced = options(10_000);
    fenced.no_code_fences = true;
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("fence", &rows), &fenced).expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert_eq!(tally.refused_code, 1);
    let row = taken[0].as_record().expect("record");
    assert_eq!(field_str(row, "id").expect("id"), "prose");

    // Off by default: the same pool converts whole.
    let (all, off) =
        convert_rows(tokenizer(), load_pool("fence_off", &rows), &options(10_000))
            .expect("converts");
    assert_eq!(all.len(), 2);
    assert_eq!(off.refused_code, 0);
}

#[test]
fn the_window_skips_a_row_that_does_not_fit() {
    let long_answer = "A sentence that repeats. ".repeat(600);
    let rows: [(&str, &[(&str, &str)]); 2] = [
        ("fits", &[("user", "Short?"), ("assistant", "Short.")]),
        ("long", &[("user", "Long?"), ("assistant", &long_answer)]),
    ];
    let mut narrow = options(10_000);
    narrow.seq_len = 256;
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("window", &rows), &narrow).expect("converts");
    assert_eq!(tally.rows_taken, 1);
    assert_eq!(tally.too_long, 1);
    let row = taken[0].as_record().expect("record");
    assert_eq!(field_str(row, "id").expect("id"), "fits");
}

#[test]
fn the_budget_walk_stops_once_crossed_and_the_last_row_overshoots() {
    let rows: [(&str, &[(&str, &str)]); 3] = [
        ("one", &[("user", "First?"), ("assistant", "A first answer with words.")]),
        ("two", &[("user", "Second?"), ("assistant", "A second answer with words.")]),
        ("three", &[("user", "Third?"), ("assistant", "A third answer with words.")]),
    ];
    let (taken, tally) =
        convert_rows(tokenizer(), load_pool("budget", &rows), &options(1)).expect("converts");
    assert_eq!(taken.len(), 1, "one row crosses a one-token budget");
    assert!(tally.supervised_tokens >= 1);

    let (all, _) =
        convert_rows(tokenizer(), load_pool("budget_all", &rows), &options(1_000_000))
            .expect("converts");
    assert_eq!(all.len(), 3, "an uncrossable budget takes the whole pool");
}
