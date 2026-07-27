# shape.rs

The convention that decides the output contract, and the first config key
addressed to both sides. `env` passes through to the model and `liquid` is
consumed by Questness and stripped; this one is read by each, which is why
it is deliberately absent from the Questness key list - stripping it would
take it from the reader it was half written for.

The default is trained rather than injected, and that is the sentence the
whole file turns on. Absent `shape`, an absent `request` and an absent
`response` all mean text, and the model knows that through training rather
than through the harness filling a record in on its behalf. So nothing
here writes a default into a config on the way out. `Default` exists
because this side has to shape a returned value and cannot read the
model's disposition to do it. Two independent holders of one default, by
design - the day they disagree is a training bug rather than a merge
conflict.

It is a convention inside `<config>` rather than a tag, which costs no
slot. That matters because the seven-tag map is locked and fully
allocated, so a channel of its own would have had to displace one.

## const SHAPE_KEY

## const REQUEST_KEY

## const RESPONSE_KEY

Public because a caller composing a config has to spell them, and a caller
that spells them itself is one that cannot be caught by a rename here.
They are the wire's vocabulary rather than ours.

## enum ShapeMember

There are three members, and `text` is not a tag. `output` and `config`
name blocks; `text` names unmarked prose, which is the checkpoint-native
protocol and the thing the grammar calls zero channels. A reader looking
for a `<text>` tag will not find one, and that absence is the point rather
than a gap - declaring `text` is declaring that no channel is involved.

`input` is deliberately not a member, and this is the boundary worth
knowing before extending the vocabulary. What the caller binds is decided
by the bindings it passes and self-describes from the value's own derived
type, which is a mechanism that predates shape and answers a different
question. A member for it would give two places authority over one block.

## const MEMBERS

The vocabulary as data, so `parse` and any future rendering read the same
list. Declaration order rather than alphabetical, because it is also the
order a record's keys come back in when several are carried.

## fn spelling

## fn parse

Parsed by searching `MEMBERS` rather than by a match, which is what makes
the constant load-bearing instead of decorative: adding a member cannot
leave the parser behind.

## struct Shape

## impl Default for Shape

## fn of

The three granularities fall back independently - no `shape`, no
`request`, no `response` - because a caller naming one direction has said
nothing about the other. Collapsing them into one check would make
declaring a response silently redeclare the request.

It takes a `Value` rather than the `Record` the caller already has inside
`prepare`, and repeats that type check. The duplication buys a function
usable from outside the turn machinery, which is what the plugin needs: it
holds a config and needs the shape before there is a turn to prepare.

An empty list is refused rather than defaulted, which is the one place
this file declines to be helpful on purpose. An absent key is a caller
saying nothing; an empty list is a caller saying to carry nothing, and
those are different statements. Substituting the default for the second
would hide a mistake that is almost certainly one.

Errors here are `Err` rather than envelope rows, unlike everything on the
answer side. A malformed shape is the caller's mistake and the caller is
code, so there is nobody to repair it - the envelope vocabulary exists for
a conversation with the model.

## fn responds_with

## fn value_of

Named for what it produces rather than for what it does to its argument,
which is the same rule that keeps a helper from colliding with a parameter
in a higher-order position.

One member comes back bare and several come back as a record. The bare
form is what makes a pipeline work: a caller declaring `[output]` gets the
value itself and can pipe it onward, where a single-key record would force
a `get` at every call site. The record is keyed by member spelling in the
declared order, so the caller's own declaration is the schema.

That supersedes two earlier cuts and is worth recording as such. A
shape-varying return with no declaration was rejected because a caller
could not know what it would get; a fixed `{output, text}` record was
rejected because it forces both halves on a caller that wanted one. The
guarantee that made the fixed record attractive survives here, because the
caller declared the shape rather than discovering it.

A shortfall is an envelope rather than an error, so a model answering off
the declared shape enters the repair loop like any other malformed
emission. That is the ordinary untrained case rather than an exceptional
one - marker routing measures dead at zero of six - so the error channel
would be the wrong home for it.

The policing is here because the policy is here. A model-emitted config is
denied unless the response declares it, and `turn` reports the block
without judging it. The conservative direction is deliberate: allowed when
declared, denied otherwise, so a model cannot widen its own contract. What
the design leaves open is the third option it names, transforming rather
than allowing or denying, and nothing here does that yet.

## fn direction

## fn missing
