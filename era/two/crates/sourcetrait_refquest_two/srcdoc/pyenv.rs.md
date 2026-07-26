# pyenv.rs

The pinned python environment's home, and the materialization of the driver
scripts into it.

The scripts are embedded in the binary and written out at run time rather than
being read from the source tree. That is what makes an installed `refquest`
self-contained: the drivers travel with the binary, so a baseline does not depend
on the checkout being present or on it being the same revision the binary was
built from.

The environment itself is deliberately not embedded and cannot be. Its pins are
ai2's hybrid-era lock - the transformers release whose in-tree hybrid model
imports without remote code, the matching torch with its bundled CUDA runtime,
vllm, the linear-attention package, triton and accelerate - and that is a
multi-gigabyte tree managed by its own tool from a committed lockfile.

## const REFQUEST_DATA_SUBDIR

## fn refquest_home

Under the data home rather than beside the source, because the environment and
the materialized scripts are generated state rather than authored material.

## fn env_python

An absent environment is an error carrying the provisioning command, not a
prompt and not an attempt to create one. Building it takes minutes and a
network, so a verb that silently started provisioning would turn a baseline run
into an install.

## fn envcheck

The drift check, and it exits non-zero on drift by design. Run it before any
campaign batch: every number this crate produces is only comparable against
earlier numbers if the environment is the same one, and the pins are the whole
reason the readings are trustworthy.

## fn materialize_pysrc

Writes only when the content differs, which keeps the modification times stable
across runs. That matters because the materialized directory is what the drivers
are actually executed from, and a spurious rewrite would make every run look like
a fresh install to anything watching the tree.

Comparing content rather than a digest is fine at this size: seven scripts of a
few kilobytes each, read once per invocation, against a subprocess that takes
seconds to start.
