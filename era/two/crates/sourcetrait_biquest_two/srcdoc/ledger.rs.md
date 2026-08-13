# ledger.rs

## fn tokenizer_ledger

The day-one instrument: Quill token counts against the checkpoint's
dolma2 tokenizer (cl100k family) over the same files. A file the
segmenter refuses drops out of BOTH sides so the ratio stays a
same-set comparison; the refused list rides the report. The
dictionary/character layer split is the health metric that explains a
ratio: symbol characters are one token each under Quill while cl100k
merges whitespace runs, so a code-heavy corpus reads high on
quill_over_cl100k long before the word layer is at fault.

The llm LibConfig profile flags supply the model dir; verify_token_map
still applies because the comparison target is the checkpoint's own
tokenizer.

## fn tokenizer_ucd

The parse-validation verb over the embedded table. Reading (Unicode
17.0.0): 159,866 assigned rows, 150,139 Word / 9,727 Symbol (25
White_Space), 1,512 fold pairs, 174 scripts, dictionary offset
160,122. The ~20 ms parse is why the table has no cached artifact -
always fresh from the embedded UCD.
