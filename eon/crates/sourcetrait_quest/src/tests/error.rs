//! Error locks: how a repair reaches a caller, and by which channel.
use crate::*;

fn rows() -> Vec<String> {
    vec![
        String::from("shape::missing at output: the answer owes a typed value"),
        String::from("shape::missing at config: the answer owes a config record"),
    ]
}

#[test]
fn a_shortfall_keeps_one_row_per_member() {
    let QuestPluginError::Envelope { rows: kept } = QuestPluginError::rows(rows()) else {
        panic!("a shortfall carries its rows");
    };
    assert_eq!(kept.len(), 2, "collected rather than first-error-only");
    assert!(
        kept[0].contains("shape::missing") && kept[0].contains("output"),
        "the kind and the member both survive: {}",
        kept[0]
    );
}

#[test]
fn every_failure_leaves_as_a_labeled_error() {
    let labelled: nu_protocol::LabeledError = QuestPluginError::rows(rows()).into();
    assert!(
        labelled.msg.contains("did not conform"),
        "the headline says what happened: {}",
        labelled.msg
    );
    let help = labelled.help.expect("the rows ride as help");
    assert!(help.contains("output"), "got {help}");
    assert!(help.contains("config"), "got {help}");
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
