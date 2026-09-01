# Off-the-shelf lossy controls

Lossy controls are kept in `<experiment>/lossy/<control>/offline/` and are
evaluated with the same gradient energy-R2, optimizer replay, final-weight, and
loss-curve measurements as research candidates. They are not mixed with
`lossless/`, whose invariant is byte-exact reconstruction.

The first planned control is `C1`: `numcodecs.BitRound` followed by Zstandard.
It is a maintained, general-purpose floating-point precision filter rather
than a gradient-specific method. Sweep retained mantissa bits only enough to
clear the D1 fidelity gate, then compare its rate and replay behavior with D1.
Its role is to establish whether temporal prediction and adaptive fallbacks
provide value beyond ordinary precision trimming plus an established codec.

Additional controls (e.g. ZFP or SZ) may be added when their maintained Python
bindings are available. Each must record package/version/configuration in its
metrics and use the same external artifact location.
