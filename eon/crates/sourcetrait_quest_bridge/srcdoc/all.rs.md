# all.rs

The api, and the reason this crate exists. The flow it holds the middle of
is `eon component -> eon bridge API <- era bridge -> era component`: an eon
component programs against what is in this file and never names an era's
library, while an era bridge implements it against that library. That is
what makes another era a swapped dependency rather than an edit to every
consumer.

Both halves of the file are that one job. The traits are what an era
implements; the data types are what the wire carries between the two ends.
They sit together because a consumer needs both to say anything, and they
sit here rather than in `wire.rs` because the wire is one transport over
this vocabulary rather than the vocabulary itself.

This file was once emptied of its traits, on the reasoning that no crate
implemented them any more. That was the implementation's absence being
cited to remove the interface, and it is recorded because the shape of the
mistake is more reusable than the fix: the era bridge holds the middle of
the flow above whatever its contents happen to be at the time.

## trait Engine

Sync and not `Send`, and both halves are forced rather than chosen. The
era-two model holds `Rc` handles in its graph cache and its prefill
scratch, so it cannot cross a thread boundary at all, and one thread must
own it for its life. Writing the bound the other way would have made the
trait unimplementable by the only engine that matters, which is the test
an API of this kind either passes or is not one.

`?Send` is therefore a statement about eras in general rather than an
accommodation of this one. An era whose model happens to be `Send` loses
nothing by satisfying a looser bound.

The chunk callback returns a bool, and that return is the cancel rather
than a courtesy. A false answer means nobody is listening any more, so an
implementation stops and reports what it managed - which lets a consumer
cancel by closing its receiving end, with no second channel and no shared
flag for the two sides to disagree about.

The callback is `&mut dyn FnMut` rather than a generic parameter, so the
trait stays object-safe-adjacent and an implementation is not
monomorphised per caller. Nothing depends on that today; what it buys is
that a future boxed engine is a possible thing rather than a rewrite.

Errors cross as `String`. A typed error would put an era's error enum in
the signature of every consumer, which is precisely the leak the API
exists to prevent - so the seam flattens them deliberately, and an era's
own mirror records where.

## trait Era

This is the generic the API provides. A consumer takes `E: Era` and names
a concrete era bridge at exactly one instantiation point, so swapping eras
is one line rather than a sweep, and nothing in the consumer can reach
past this into an era's library because there is no path from here to one.

The engine is built rather than handed over, which is what the associated
type plus a returning function buys. A `FnOnce` returning a not-`Send`
value is itself `Send` while the value is not, so `engine` can cross to
the thread that will own its result. An `Era` carrying an already-built
engine could not be moved to that thread at all.

Both functions are associated rather than methods, because an era is an
identity rather than an object: there is nothing for an instance to carry,
and requiring one would make a consumer construct a value whose only
purpose is to be named.

`info` answers without loading, which is what lets a consumer say what it
is about to open before paying for the model. Its answer and
`Engine::open`'s can legitimately differ - one is the era's constant and
the other is what actually loaded - and a configuration pointing at
another checkpoint is exactly that case.

## struct EraInfo

## struct ChatOptions

The daemon does not consult these, and that is a property of the singleton
rather than an unfinished path. One container owns one model for the
service's life, so a session cannot select a checkpoint or a settings
profile; what it can do is learn which one it got, which is what `EraInfo`
answers.

A decode budget is the field that reads like it should still work and does
not. It could only be stored on the shared engine, where one session would
silently change another's. Threading it per turn is the shape that would
work, and that is a wire change rather than something to fake here.

The fields stay because the wire's `OpenRequest` carries them and because
the question they ask is the right one; what changed is who can answer it.

## struct TurnReport

## enum FinishReason

`Cancelled` is a first-class ending rather than an error, which is what
lets a consumer report a stopped turn in the same place it reports a
finished one. On the daemon side a cancel arrives as a closed chunk
channel, so the engine stops and still produces a report - the accounting
is real rather than synthesised.
