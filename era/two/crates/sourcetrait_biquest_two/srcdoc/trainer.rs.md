# trainer.rs

The BiquestTrainer: the organism's fresh checkpoint (init) and the
full-parameter training verb (train) over the llm lib's
imagine_quest_train module.

## struct OrganismSpec

The geometry derivation, every ratio the hybrid's own scaled to the
organism width: intermediate at 2.75x hidden (rounded clean at the
multiple-of-256 widths the flag produces), GDN key/value head dims at
3/4 and 3/2 of the attention head dim (the hybrid's 96/192 against
128), conv kernel 4, the 3:1 GDN-to-attention layer cycle (G G G A).
The multiple-of-four layer floor keeps the validator's
genuine-hybrid requirement (both kinds present) true by construction.

Design fills recorded here rather than ruled: negative eigenvalues
allowed (the gated-delta convention, beta in (0,2)); NoPE attention
(no rope_parameters block, matching the hybrid's full-attention
layers); eos and pad ids at NULL (0x00) until the organism's stop
story is designed - generation against the organism must override
stops until then.

## the GDN gate inits

A_log seeds ln(U(1,16)) and dt_bias seeds softplus-inverse of
log-uniform dt in (0.001, 0.1) - the Mamba-family convention the
gated delta rule inherits. These are the two tensors where a plain
gaussian would be wrong: decay must start spread across the useful
range, not clustered at exp(-exp(0)).

## fn trainer_init

The embedding passes through as raw bf16 bytes - no decode-encode
round trip, so the checkpoint's embedding is bit-identical to the
matrix artifact's. The tied head writes the same bytes under
lm_head.weight because the candle engine reads that name
unconditionally. The acceptance check is the llm loader itself:
`load_config` parses and policy-validates the written config.json
before the verb returns, so an organism checkpoint that would refuse
to load cannot be produced silently.

## fn pack_corpus

Plain LM packing: one id stream in file-walk order, sliced at stride
seq_len into chunks of seq_len + 1, so the boundary token closes one
chunk as target and opens the next as input and no position is
wasted. No separator is injected between files - the stop story and
any document-boundary convention are undesigned, and inventing one
here would bake it in silently; the files' own trailing newlines are
the only seam. A refused file fails the whole pack (the ledger
skips-and-reports because it is an instrument; a trainer that skips
content trains on a corpus nobody chose). Unconditionally compiled
so the packing locks run in the default build.

## fn trainer_train

The verb parses in every build; the non-train build raises the
one-sentence rebuild message (the bquest convention - the parser is
feature-blind so `doc cli` renders one tree). Backend by cfg:
train-cuda takes the cuda autodiff pair, plain train the cpu pair.
The id-space guard (`vocab >= keywords + characters + rows + words`)
catches a checkpoint/word-file mismatch where the word count shrank;
a same-size swap is undetectable here and remains the operator's
alignment to keep. The step log streams as NUON lines through the
associations module's condensed renderer; mid-run checkpoints land
under `<out>/steps/step_N/`; the final checkpoint, provenance, and
log all land in `<out>` itself.
