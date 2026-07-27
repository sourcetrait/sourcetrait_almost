//! Channel-grammar locks: the slot map's id assignment, the
//! line-anchored framing round trip, newline escaping, and the
//! tool-marker aliasing the boundary carries until routing is trained.
use crate::*;
use crate::channel::{
    Aliasing,
    Block,
    Envelope,
    TAGS,
    Tag,
    escape_content,
    parse_blocks,
    render_block,
    render_blocks,
    unescape_content,
};

#[test]
fn the_slot_map_is_the_contiguous_reserved_run() {
    assert_eq!(Tag::Config.token_id(), 100_256);
    let reserved_run: Vec<u32> = TAGS
        .into_iter()
        .filter(|tag| *tag != Tag::Config)
        .map(|tag| tag.token_id())
        .collect();
    assert_eq!(
        reserved_run,
        vec![100_270, 100_271, 100_272, 100_273, 100_274, 100_275]
    );
    // The closer is pinned at the far end so the openers stay
    // contiguous; that is what let a seventh tag extend the run.
    assert_eq!(Tag::Close.token_id(), 100_275);
    assert!(!Tag::Close.opens());
    assert_eq!(TAGS.into_iter().filter(Tag::opens).count(), 6);
}

#[test]
fn every_tag_carries_a_distinct_id_spelling_and_name() {
    let mut ids: Vec<u32> = TAGS.into_iter().map(|tag| tag.token_id()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), TAGS.len());
    for tag in TAGS {
        assert!(tag.spelling().starts_with("<|extra_id_"), "{tag:?}");
        assert!(!tag.name().is_empty());
    }
}

#[test]
fn a_block_round_trips_through_the_line_anchored_form() {
    let block = Block::new(Tag::Output, "list<string>", "[Var, bar, Car]");
    let text = render_block(&block);
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("<|extra_id_2|> list<string>"));
    assert_eq!(lines.next(), Some("[Var, bar, Car]"));
    assert_eq!(lines.next(), Some("<|extra_id_6|>"));
    let parsed = parse_blocks(&text, Aliasing::Strict).expect("parses");
    assert_eq!(parsed, vec![block]);
}

#[test]
fn pass_renders_inline_and_parses_back() {
    let block = Block::new(Tag::Pass, "", "$args");
    let text = render_block(&block);
    assert_eq!(text, "<|extra_id_3|>$args<|extra_id_6|>");
    assert_eq!(text.lines().count(), 1);
    assert_eq!(parse_blocks(&text, Aliasing::Strict).expect("parses"), vec![block]);
}

#[test]
fn several_blocks_round_trip_in_emission_order() {
    let blocks = vec![
        Block::new(Tag::Output, "list<string>", "[Var, bar, Car]"),
        Block::new(Tag::Pass, "", "$args"),
        Block::new(Tag::Output, "record<name: string>", "{name: foo}"),
        Block::new(Tag::Pass, "", "$in"),
        Block::new(Tag::Nu, "def evaluate []: nothing -> nothing {", "}"),
    ];
    let text = render_blocks(&blocks);
    assert_eq!(parse_blocks(&text, Aliasing::Strict).expect("parses"), blocks);
}

#[test]
fn an_empty_content_block_round_trips() {
    let block = Block::new(Tag::Liquid, "", "");
    let text = render_block(&block);
    assert_eq!(text, "<|extra_id_5|>\n<|extra_id_6|>");
    assert_eq!(parse_blocks(&text, Aliasing::Strict).expect("parses"), vec![block]);
}

#[test]
fn escaping_is_an_exact_inverse_over_backslashes_and_breaks() {
    // A literal backslash-n must survive beside a real newline, which
    // is the whole reason the escape doubles backslashes first.
    let raw = "a\\nb\nc\r\nd\\\\e";
    let escaped = escape_content(raw);
    assert!(!escaped.contains('\n'), "escaped content is single-line");
    assert!(!escaped.contains('\r'), "escaped content carries no return");
    assert_eq!(unescape_content(&escaped), raw);
}

#[test]
fn an_escaped_payload_cannot_forge_a_boundary() {
    let forged = format!("{{note: \"x\n{}\"}}", Tag::Close.spelling());
    let block = Block::new(Tag::Output, "record<note: string>", &escape_content(&forged));
    let text = render_block(&block);
    let parsed = parse_blocks(&text, Aliasing::Strict).expect("parses");
    assert_eq!(parsed.len(), 1, "the payload must not split the block");
    assert_eq!(unescape_content(&parsed[0].content), forged);
}

#[test]
fn an_unclosed_block_is_an_error_naming_its_tag() {
    let text = "<|extra_id_2|> list<string>\n[a, b]";
    let error = parse_blocks(text, Aliasing::Strict)
        .expect_err("unclosed rejects")
        .to_string();
    assert!(error.contains("<output>"), "got {error}");
}

#[test]
fn strict_refuses_unmarked_content() {
    let text = "here is some prose\n<|extra_id_2|> int\n4\n<|extra_id_6|>";
    assert!(parse_blocks(text, Aliasing::Strict).is_err());
    let trailing = "<|extra_id_2|> int\n4\n<|extra_id_6|>\nand a trailing remark";
    assert!(parse_blocks(trailing, Aliasing::Strict).is_err());
}

#[test]
fn the_boundary_aliases_the_trained_attractor_to_output() {
    // Zero of 63 probe generations opened with the output marker;
    // every structured emission routed through function_calls instead.
    let text = "<function_calls>\n{name: foo}\n</function_calls>";
    let parsed = parse_blocks(text, Aliasing::ToolMarkers).expect("aliases");
    assert_eq!(parsed, vec![Block::new(Tag::Output, "", "{name: foo}")]);
    assert!(parse_blocks(text, Aliasing::Strict).is_err());
}

#[test]
fn the_boundary_aliases_an_unmarked_body_to_output() {
    let text = "{name: foo}";
    let parsed = parse_blocks(text, Aliasing::ToolMarkers).expect("aliases");
    assert_eq!(parsed, vec![Block::new(Tag::Output, "", "{name: foo}")]);
}

#[test]
fn the_diagnostic_envelope_renders_as_a_nuon_record() {
    let mut envelope = Envelope::default();
    assert!(envelope.is_clean());
    envelope.error("channel::conform", Some("$.rows.0.name"), "expected string");
    envelope.warn("channel::extra_field", None, "unknown key carried through");
    assert!(!envelope.is_clean());
    let text = nu::to_nuon_text(&envelope.to_value()).expect("renders");
    assert!(!text.contains('\n'), "one line, got {text}");
    let ty = nu::parse_typedef(
        "record<errors: table<kind: string, source: oneof<string, nothing>, \
         message: string>, warnings: table<kind: string, source: \
         oneof<string, nothing>, message: string>>",
    )
    .expect("typedef");
    nu::conform(&envelope.to_value(), &ty).expect("the envelope conforms");
}
