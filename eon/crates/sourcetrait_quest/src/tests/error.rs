//! Error locks: how a repair reaches a caller, and by which channel.
use crate::*;

fn envelope() -> lib::channel::Envelope {
    let mut envelope = lib::channel::Envelope::default();
    envelope.error(
        "shape::missing",
        Some("output"),
        "the response shape declares output, so the answer owes a typed value",
    );
    envelope.error("channel::parse", None, "unmarked content outside every block");
    envelope
}

#[test]
fn a_repair_envelope_keeps_one_row_per_diagnostic() {
    let QuestPluginError::Envelope { rows } = QuestPluginError::envelope(&envelope()) else {
        panic!("an envelope carries its rows");
    };
    assert_eq!(rows.len(), 2, "collected rather than first-error-only");
    assert!(
        rows[0].contains("shape::missing") && rows[0].contains("output"),
        "the kind and its source both survive: {}",
        rows[0]
    );
    assert!(
        !rows[1].contains(" at "),
        "a row with no source does not invent one: {}",
        rows[1]
    );
}

#[test]
fn every_failure_leaves_as_a_labeled_error() {
    let labelled: nu_protocol::LabeledError =
        QuestPluginError::envelope(&envelope()).into();
    assert!(
        labelled.msg.contains("did not conform"),
        "the headline says what happened: {}",
        labelled.msg
    );
    let help = labelled.help.expect("the rows ride as help");
    assert!(help.contains("shape::missing"), "got {help}");
    assert!(help.contains("channel::parse"), "got {help}");
}

#[test]
fn an_insufficiency_is_reported_as_the_model_declining() {
    let labelled: nu_protocol::LabeledError = QuestPluginError::Insufficient.into();
    assert!(
        labelled.msg.contains("could not answer"),
        "declining is not a harness fault: {}",
        labelled.msg
    );
    assert!(labelled.help.is_none(), "there are no rows to carry");
}
