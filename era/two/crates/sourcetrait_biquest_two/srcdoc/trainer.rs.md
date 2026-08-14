# trainer.rs

The BiquestTrainer's init half: the organism's complete fresh
checkpoint, so the ImagineQuestMatrix artifact becomes a model the
llm engine can actually load. The training loop is the module's
other half, not yet landed.

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
