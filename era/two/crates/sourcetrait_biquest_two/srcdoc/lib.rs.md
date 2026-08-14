# lib.rs

The manifest. biquest is the QuestImaginary fork of bquest: the hybrid-era
machinery (capability harness, checker ports, mix pipeline, theirs
conversion, taskgen, speculation probe, adapter train verbs) is removed
rather than maintained, because the fork tracks nothing and owes no
compatibility. What survived the cut: the CLI skeleton, the error type,
the nu-value bridge (`value.rs`), and the self-documentation verb. What
is new: the ImagineQuestTokenizer modules (`ucd`, `lexer`,
`dictionary`, `ledger`, with `corpus` as the shared file walker), the
ImagineQuestAssociations store (`associations`), and the
WikimediaDumpTool (`wikimedia` for the verbs, `wikixml` for the XML
page layer over the raw dumps).

The llm and harness deps stay: the harness carries the nu data core the
artifacts ride on, and the llm lib carries the model core plus the
imagine_quest_train module the trainer train verb drives. The train
features forward to the llm lib; trainer_train is the one cfg-gated
dispatch (a non-train build raises the one-sentence rebuild message).
