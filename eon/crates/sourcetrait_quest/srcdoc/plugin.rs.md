# plugin.rs

## type Harness

## struct QuestPlugin

Two things are held across calls and both earn it. A tokio runtime and
the evaluator's base engine are each built once and reused, and the
plugin process lives for a burst of calls rather than forever - stock
garbage collection tears it down after an idle timeout, so residency is
bounded without anything here arranging it.

That lifecycle is the reason nothing else is cached. A session, a
snapshot or a warmed context would outlive its usefulness and then be
collected mid-life; residency for those is state-scoped and belongs to
whatever holds the state.

## fn new

The tool markers stay aliased, and that is a measured accommodation
rather than a default anyone chose. The channel probe reads zero of six
on the marker the grammar wants, so the boundary has to accept what the
model actually emits. Training retires it, and the alias is named for
that reason rather than folded into a general leniency.

## fn runtime

## fn think_harness

A poisoned lock is recovered rather than propagated. One call panicking
must not make every later call fail, and there is no invariant behind the
lock to protect: what is there is an engine and a world, and a fresh turn
builds its own state over both.

## impl Plugin for QuestPlugin

`version` reports the crate's own, which is what nushell records in its
registry and compares on a later load. It is deliberately the crate
version rather than the protocol's: the registry caches signatures and
never self-refreshes, so a version that moves when the surface moves is
what makes a stale registration visible.
