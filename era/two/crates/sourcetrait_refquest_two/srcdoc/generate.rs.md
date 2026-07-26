# generate.rs

The generation verb: build the driver's argument vector and hand it to the shared
plumbing.

Every argument is passed explicitly rather than defaulted on the python side, so
a recipe is fully described by the command line that ran it. That is what makes a
baseline reproducible from its own record.

## fn generate

Two arguments are derived rather than passed through, and both are correctness
rather than convenience.

The fla blocker is forced whenever the device is CPU, regardless of the flag.
fla's gated norm constructs against the current CUDA device at model
initialisation, so a CPU run with fla importable touches CUDA and crashes at init
rather than running on the CPU.

The GPU is hidden as well, by an empty device list in the environment. Blocking
fla alone is not enough to make a CPU grade genuinely CUDA-free, and the reason
to want that is measurement rather than tidiness: the CPU grade's card delta must
read exactly zero for it to be evidence about the CPU path at all.

Sampling arguments are only passed when sampling is on, which keeps a greedy
command line free of values that would not be read. Stop tokens are repeatable
and passed through as given, because the shipped generation config is bare - it
carries the end-of-sequence token and nothing else - so the driver supplies them
explicitly in chat mode rather than inheriting a recommended posture that does
not exist in the file.
