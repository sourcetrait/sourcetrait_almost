# lib.rs

The manifest. biquest is the QuestImaginary fork of bquest: the hybrid-era
machinery (capability harness, checker ports, mix pipeline, theirs
conversion, taskgen, speculation probe, adapter train verbs) is removed
rather than maintained, because the fork tracks nothing and owes no
compatibility. What survived the cut: the CLI skeleton, the error type,
the nu-value bridge (`value.rs`), and the self-documentation verb. What
is new: the ImagineQuestTokenizer modules (`ucd`, `lexer`,
`dictionary`, `census`, `ledger`) and the ImagineQuestAssociations
store (`associations`).

The llm and harness deps stay: the harness carries the nu data core the
artifacts ride on, and the llm lib carries the model core the
BiquestTrainer task will train from scratch. The train features forward
to the llm lib for that later task; nothing in this crate is gated on
them yet.
