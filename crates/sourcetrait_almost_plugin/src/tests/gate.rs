use crate::*;

#[test]
fn check_reports_the_stage_taxonomy() {
    assert_eq!(
        check("record<quiet: bool>", "{quiet: true}").stage,
        CheckStage::Pass
    );
    assert_eq!(check("record<", "{}").stage, CheckStage::TypedefParse);
    assert_eq!(
        check("record<quiet: bool>", "{quiet:").stage,
        CheckStage::NuonParse
    );
    assert_eq!(
        check("record<quiet: bool>", "{quiet: 3}").stage,
        CheckStage::Conformance
    );
}

#[test]
fn check_carries_declared_and_derived_renderings() {
    let report = check("record<quiet: bool>", "{quiet: 3}");
    assert_eq!(report.declared, "record<quiet: bool>");
    assert_eq!(report.derived.as_deref(), Some("record<quiet: int>"));
    assert!(report.error.is_none());
}
