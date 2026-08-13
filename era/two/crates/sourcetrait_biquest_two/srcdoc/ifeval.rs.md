# ifeval.rs

The IFBench instruction verifiers plus olmo-eval's strict and loose scoring
harness, ported checker by checker from `instructions.py`.

Classic Google-IFEval ids are not ported and raise loudly rather than returning
false. The exercised capability set is the out-of-distribution one, and a silent
mis-score would be worse than a loud gap - a false reads as a model failure where a
raise reads as a missing checker.

Several reference quirks are preserved deliberately, because matching means
matching rather than improving. They are called out at their own checkers below,
and each one is a place where fixing the bug would move a score away from the
reference rather than toward the truth.

What these actually measure is worth holding while reading them, because the type
names mislead. A minority test conformance to a stated output constraint, which is
the job. The majority are constrained-writing puzzles - palindromes, prime-length
words, alternating syllable counts, alphabetical sentence openings. And a handful
have their expected answer baked into the checker, so they measure recall of one
specific prompt rather than any capability.

## enum KwargValue

## fn as_f64

## fn as_index

## fn as_text

The integer and float split is semantic rather than a transport artefact: an index
must be an integer while a threshold may be either. A float used as an index raises
rather than truncating, since truncation would silently shift a position.

## type Kwargs

## fn kwarg

A missing kwarg raises, and the message says the reference would have randomised.
That is the deliberate divergence: choosing determinism over silent variation, since
a randomised argument makes a score unreproducible.

## fn loose_variants

The eight response variants in upstream's order: the response itself, asterisks
stripped, the first line removed, the last line removed, both removed, and those
three again with asterisks stripped. Loose scoring passes if any variant passes.

The slice arithmetic is guarded against short inputs rather than assuming at least
two lines, since a single-line response reaches here.

## fn check_one

The prompt-to-repeat injection, the non-empty gate and then the verifier, in
upstream's order. An empty response is false before any checker sees it, which is
what stops a checker whose condition is vacuously true on empty text from passing.

## fn score_ifeval_item

## const ASCII_LOWER

## fn count_words

## fn first_chars_equal

## macro_rules checker_regex

## static DIGIT_RUNS

## static JAPANESE

## static DATE_YMD

## static MCQ_SPLIT

## static MCQ_OPTION

## static CSV_HEADER_CITY

## static CSV_HEADER_QUOTES

## static CSV_SPECIAL_FIELD

Python's whitespace class is spelled out inline in the patterns that need it rather
than interpolated, because these are raw strings carrying their own escapes.

## fn py_csv_parse

Python's `csv.reader` as a state machine, at its default quoting with doublequote
and non-strict parsing. Records end at a newline, a carriage return, or the pair,
outside quotes.

The non-strict branch is the one that matters: a character following a closing quote
mid-field joins the field rather than raising, which is what Python does and what a
strict parser would reject. Model output hits that case.

## fn check_following

The verifier dispatch, one arm per instruction id. Four arms are worth reading for
the reference quirks they reproduce rather than fix.

The conjunction counter filters on the cleaned word but collects the raw one into its
set, so distinct casings and punctuation variants count separately - which makes the
threshold easier to clear than the type name suggests.

The line-indent checker removes blank lines with a mutate-while-iterating loop, and
the skips that produces are reproduced rather than corrected: the reference's loop
advances its index after a removal, so it steps over the line that shifted into
place.

The sentence-words ratio compares character length while requiring exactly three
sentences, which its name does not say at all.

The title-case checker passes an all-caps word, because its cases test for
upper-then-lower and lower-then-upper and an all-caps rest falls through both.

Two more are structural rather than quirks. The options checker derives its
separator by looking for a slash, then the word "or", then a comma, which means an
option list containing the letters "or" inside a word takes the wrong split. And the
emoji checker looks at the last two characters and then at the next sentence's
first, so an emoji between sentences satisfies the sentence before it.

## fn treebank_word_tokens_without_punctuation
