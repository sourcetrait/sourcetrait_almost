# chat.rs

Byte-exact to the checkpoint's `chat_template.jinja`, which is byte-identical
to era one's at the same sha and the same 2613 bytes - the one sanctioned code
carry across eras.

Two surfaces sit here for two callers. `chat_wrap` and `chat_continue` are the
single-shape inference helpers the engine and the battery rigs drive.
`chat_render` is the message-list form the trainer needs, returning the text
AND the byte span of every assistant turn so a training example's loss boundary
is exact by construction from ONE render.

The span form exists because the alternative is measurably wrong against THIS
template. An assistant turn renders differently BY POSITION - the end-of-text
token when last, `<|im_end|>` plus a newline when interior - so the upstream
practice of re-rendering the conversation prefix and taking its token count
measures one token short, once per interior assistant turn. One render plus
spans has no positional invariant to maintain and costs one pass instead of
n+1.

A span COVERS its turn's terminator and STOPS there. The newline after an
interior `<|im_end|>` opens the NEXT turn rather than closing this one, and our
harness re-renders the whole conversation each turn, so the model is never
asked to produce it. That exclusion is a decision rather than the inherited
off-by-one.

## const INJECTED_SYSTEM

## fn chat_wrap

## fn chat_continue

Every post-decode save ends mid assistant turn, because the stop token is
sampled and never consumed into the caches. So a continuation has to close that
open turn before it can open a user turn, which is why this exists as a
separate rendering rather than as `chat_render` over a message list.

## fn resolve_stop_ids

## enum ChatRole

`Tool` is the template's own alias for `Environment` and renders identically.
It exists so foreign data carrying that role round-trips without the caller
rewriting it, not because the two mean different things here.

### fn rendered

### fn parse

An unknown role raises rather than defaulting, because training data carrying a
role we silently dropped would train against a render nobody inspected.

## struct ChatMessage

`content` is optional rather than defaulted because the template tests PRESENCE
rather than truthiness: absent renders nothing while an EMPTY string renders an
empty body. Both engines agree on that, verified against the reference - jinja
tests `is not none`, so Liquid's Ruby truthiness never diverges there.

`functions` and `function_calls` are the template's STRING paths, which is what
the checkpoint's own data populates. The structured `tools` argument reaches a
different branch that is dead for every path this crate takes.

### fn new

### fn system

### fn user

### fn assistant

### fn environment

### fn is_role

## struct AssistantSpan

### fn len

### fn is_empty

## struct ChatRender

A trailing generation prompt produces no span, because there is no content
behind it yet.

## fn chat_render

`add_generation_prompt` appends the assistant opener after the final message
whatever its role, which is the template's own behavior rather than a
convenience.

## struct TokenSpan

## struct EncodedRender

## fn encode_render

A span boundary that does not coincide with a token boundary is a HARD ERROR
rather than a rounded one. The mask it feeds decides which positions carry
training signal, so a silently shifted boundary trains the wrong thing with no
symptom anywhere.

The whole render encodes as ONE string, matching how the same text is encoded
at inference. Encoding piecewise would differ across a BPE boundary.

The boundary map keys the FIRST token starting at each byte offset, plus an
end-of-text sentinel at the render's length, which is what makes a span ending
exactly at the render's end resolvable.
