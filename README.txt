A beginning is worth one limit: robust unlearnable laws in noisy cellular automata
Seth Douglas and Nidhal Mghirbi

Version 1.0.0, October 2026. A preprint; it has not been peer reviewed.

We compare nested expectation laws of begun and stationary probabilistic cellular automata. At equal scale depth d, a computable begun law has ceiling Delta^0_(d+2), whereas consensus on a nonempty effectively Sigma^0_2 class of stationary translation-invariant states has ceiling Delta^0_(d+1). We attain the begun ceiling at every finite depth using positive-rate three-dimensional rules, an open kernel region and a uniform renewal-gap readout schema. The canonical seed clones a uniform Bernoulli row on the physical block alphabet. Seed universality refers to blocked full-support, one-cell-shift-invariant ergodic laws on the pre-blocking alphabet. The seed has no distinguished origin or program but retains infinite transverse redundancy. At depth one, local error certificates and a continuous plateau quantizer give every Delta^0_3 value in [0,1] with a clock-free readout and genuine inner time limits. Every ordinary and Cesaro cluster state carries the same law; at computable kernels its stationary Cesaro cluster class is Pi^0_2 and has no low member when the value is outside Delta^0_2. An exact conditional-cone identity transfers any computable-seed begun depth-d law to a depth-(d+1) law in every stationary state of the same positive-rate rule. The extra limit is sufficient and sharp across the arithmetic ranges, but need not be minimal for an individual law. In the normalized four-dimensional Gibbs formulation its lag is a slab separation. At computable begun data, fixed targets outside Delta^0_2 cannot be learned with positive probability by effective finite-history inquiries. Removing cloned preparation from the realization remains open.

The manuscript and source are in paper/. The self-contained submission source
is paper/arxiv/. Sequential finite checks, recorded outputs and LF checksums
are in paper/anc/.

Build outside the source tree:
    python paper/build.py --output-dir BUILD_DIRECTORY
With Python, NumPy, psutil and threadpoolctl, verify the pristine
distribution before rerunning checks:
    python paper/anc/verify_hashes.py
The programs write beside themselves. Copy the complete paper/anc/ directory
to a new working directory, then run the copy:
    python WORKING_DIRECTORY/run_checks.py
Reruns produce new timing and memory values and are not expected to match
the distribution's SHA256SUMS. Preserve the pristine distribution and its
manifest for archive verification.

The strict builder runs pdflatex and BibTeX, settles references and rejects
warnings, undefined references, missing glyphs and bad boxes. No new DOI has
been assigned to this preprint.

Here A is the pre-blocking alphabet and Sigma=A^B is the physical block
alphabet. Universality means clones of blocked full-support, one-cell-shift-
invariant ergodic A-row laws; the uniform Bernoulli Sigma-row is included.

The inquiry conclusion cites Uncomputable critical exponents, version 2.1.0,
Theorems 21.1 and 1.5: https://doi.org/10.5281/zenodo.23780285.
Realization and consensus do not depend on the companion inquiry theorem.

The cloned seed, effective stationary consensus class, uniform readout schema
and prescribed limit order are part of the claims. Transfer adds one sufficient
limit; no universal minimal cost or necessity of an internal clock is claimed.
Finite simulations do not certify noncomputability or all-time reliability.

The text and documentation are CC BY 4.0; scripts are MIT or Apache 2.0 at your
option. See RIGHTS.md and the license files.
