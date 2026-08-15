//! Store locks: the concept table, gloss sanitizing, and the
//! single-line NUON rendering guard. The dump-side absorption locks
//! live with the derivation (tests/wikiderive).
use crate::associations::condensed_line;
use crate::associations::sanitize_gloss;
use crate::associations::AbstractConcept;
use crate::lexer::KEYWORD_REPETITION;
use crate::value::v_str;

#[test]
fn concepts_are_backed_by_their_markers() {
    assert_eq!(AbstractConcept::ALL.len(), 2);
    let repetition = AbstractConcept::Repetition;
    assert_eq!(repetition.name(), "repetition");
    assert_eq!(repetition.marker(), KEYWORD_REPETITION);
    assert_eq!(repetition.spelling(), "<|repetition|>");
    let case = AbstractConcept::Case;
    assert_eq!(case.name(), "case");
    assert_eq!(case.marker(), crate::lexer::KEYWORD_CASE);
    assert_eq!(case.spelling(), "<|case|>");
}

#[test]
fn glosses_sanitize_to_single_line_single_spaced() {
    assert_eq!(sanitize_gloss("  a\n\tb   c\r\n"), "a b c");
    assert_eq!(sanitize_gloss("\n \t"), "");
}

#[test]
fn condensed_rows_render_single_line_and_raw_newlines_refuse() {
    let engine_state = nu_protocol::engine::EngineState::new();
    let clean = condensed_line(&engine_state, &v_str("a \"quoted\" gloss")).expect("renders");
    assert!(!clean.contains('\n'));
    condensed_line(&engine_state, &v_str("two\nlines")).expect_err("multi-line rows refuse");
}
