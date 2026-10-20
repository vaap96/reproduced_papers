# Action Required: Send to MerLin Team

## Summary

No MerLin defect was found. The photonic adaptation ran without MerLin errors
on CPU and on an NVIDIA L4 GPU, and agreed with independent references. Five
points could help future users: the 0.2.x vs 0.4.x measurement API gap, how to
get per-mode threshold-detector outcomes from a `QuantumLayer`, the undocumented
device-memory overhead of `FOCK` simulation (about 22× the output), a modest GPU
speed-up over CPU, and size guidance for the `FOCK` computation space.

## Context

The reproduction covers *Power of data in quantum machine learning* (Huang et
al., Nat. Commun. 2021), a study of quantum kernels (fidelity and projected)
against tuned classical models. The paper has no photonic experiments, so the
reproduction adds a simulated MerLin adaptation (translation status
`MERLIN_IMPLEMENTATION`, `lib/photonic_merlin.py`): two linear-optics embeddings,
a fidelity kernel from `AMPLITUDES`, a projected kernel on per-mode no-click
probabilities, and labels from `PROBABILITIES`. It ran with MerLin 0.2.3 and
Perceval for 10 seeds: n = 4–14 on an Apple M5 CPU (float64), and up to 18 modes
with 9 photons (Fock dimension 3.1 M) and 21 modes with 8 photons on an NVIDIA
L4 (float32); 560 records plus a 20-point scaling probe. The
observations below come from that work and are independent of its science.

## Items to Send

### Medium — Documentation: cookbook targets 0.4.x while 0.2.3 is in use

- Type: missing documentation
- What happened: the cookbook snippets target MerLin 0.4.1, where measurement
  strategies are factory methods. In MerLin 0.2.3 (the version in this
  repository's shared `.venv` and pinned by this paper) `MeasurementStrategy` is
  an enum with the members `AMPLITUDES`, `MODE_EXPECTATIONS` and
  `PROBABILITIES`, so the snippets do not run as written. Papers in this
  repository pin different versions (0.2.3 and 0.4.0), so one environment
  cannot serve all of them. The adaptation had to be written against the 0.2.x
  API after inspecting it directly.
- Evidence:
  - `papers/power_of_data_qml/LOG.md`, "Translation paths assessed (MerLin
    0.2.3)": "the cookbook targets 0.4.1, whose measurement API differs (enum
    in 0.2.3)".
  - `papers/power_of_data_qml/FEEDBACK.md`, "MerLin integration feedback".
  - Check: `python -c "import merlin as ml; print(list(ml.MeasurementStrategy))"`
    on 0.2.3 lists the three enum members.
  - Not tested: the 0.4.x API itself; the 0.4.x behaviour is taken from the
    cookbook text.
- Why it matters: anyone reproducing a paper in this repository with a
  different pinned MerLin version will hit the mismatch, and `lib/photonic_merlin.py`
  needs a port if the environment moves to 0.4.x.
- Suggested action: add a version-compatibility note to the cookbook (a table
  of the measurement API per MerLin version) and a short 0.2.x → 0.4.x
  migration note for `QuantumLayer` measurement strategies.

### Medium — QuantumLayer: how to obtain per-mode threshold-detector outcomes

- Type: API confusion (possible documentation gap; not confirmed as a missing
  feature)
- What happened: the projected kernel needs per-mode no-click probabilities
  P(n_k = 0), the threshold-detector counterpart of a one-qubit reduced state.
  We computed them by hand from the `FOCK` probability vector. MerLin 0.2.3
  also contains `merlin.DetectorTransform` (per-mode Perceval detectors with
  `partial_measurement`), but `QuantumLayer.__init__` has no `detectors`
  argument and we did not find how to connect the two, so we did not use it. We
  did not test whether it would reproduce our features.
- Evidence:
  - `papers/power_of_data_qml/lib/photonic_merlin.py` (hand-written no-click
    features, with `FOCK` computation space and `AMPLITUDES` / `PROBABILITIES`).
  - Signature check on 0.2.3: `QuantumLayer(input_size, builder, circuit,
    experiment, input_state, n_photons, trainable_parameters, input_parameters,
    amplitude_encoding, computation_space, measurement_strategy, device, dtype,
    **kwargs)`; `DetectorTransform(simulation_keys, detectors, *, dtype, device,
    partial_measurement)`.
  - Cross-checks that did pass: `UNBUNCHED` probabilities equal `FOCK`
    probabilities postselected on collision-free outcomes and renormalized (max
    |difference| = 0.0), and `FOCK` probabilities equal |permanent|² (0.047215
    both ways).
  - Related prior work in this repository: `papers/Bandwidth_tunning/lib/kernels.py::ProjectedFidelityKernel`
    (MerLin 0.4.0) builds an RBF on per-mode zero-photon marginals. We did not
    compare its approach with ours.
  - Confidence: low on whether a supported path exists; high on the signatures.
- Why it matters: threshold detection is how many photonic devices measure, and
  a no-click marginal without postselection needs the `FOCK` space. Users who
  want it must currently rebuild it from the full probability vector, which is
  easy to get wrong.
- Suggested action: document (or add an example for) per-mode threshold
  detection on a `QuantumLayer`, say whether `experiment` with detectors or
  `DetectorTransform` is the intended route, and note the relation between
  `UNBUNCHED` and postselected `FOCK`.

### Medium — QuantumLayer (FOCK, CUDA): device memory is about 22× the output

- Type: missing documentation (possible improvement)
- What happened: a `QuantumLayer` with `ComputationSpace.FOCK`, `AMPLITUDES` or
  `PROBABILITIES`, `dtype=torch.float32` on `cuda:0` peaks at about 175 B per
  complex64 output amplitude for a batch (about 22× the 8 B of the output),
  plus about 3 KB per Fock state independent of the batch. Example: 200 points
  × 230,230 states peaked at 8,039 MB; 8 points × 3,124,550 states at
  14,134 MB. Sizing batches on the output tensor ran out of memory at 490,314
  states (18 GB requested on a 23 GB L4). With batches sized for the 22× peak,
  every point up to 3.1 M states ran.
- Evidence:
  - `papers/power_of_data_qml/LOG.md`, "Probe attempts" (attempt 2, out of
    memory) and "Probe result (attempt 3)"; probe runs
    `outdir/run_20261019-160241` and `outdir/run_20261019-160710`
    (`metrics.json`, `peak_device_mb` from `torch.cuda.max_memory_allocated`).
  - `papers/power_of_data_qml/results/photonic_scaling.md` (20 probe points and
    the production runs, which match the probe).
  - MerLin 0.2.3, Perceval 1.3.1, torch 2.14.1+cu130, NVIDIA L4; observed in
    every probe point and production unit. The internal cause was not
    investigated (no profiling of MerLin internals).
- Why it matters: users sizing GPU batches from the output shape will hit
  out-of-memory errors about 20× earlier than expected, and the overhead caps
  the reachable Fock dimension per batch.
- Suggested action: document the expected peak memory per batch for `FOCK`
  simulation, or expose a memory estimate; if the 22× factor comes from
  avoidable intermediates, reducing it would let larger batches run.

### Low — Performance note: GPU speed-up over CPU was modest at 10^5 states

- Type: other (observation)
- What happened: for 800-point banks at Fock dimensions 77,520 and 116,280 the
  L4 (float32) took 4.9 s and 7.4 s against 8.6 s and 12.0 s on an Apple M5 CPU
  (float64), a 1.6–1.8× speed-up. Simulation time was close to linear in the
  Fock dimension (about 75–90 µs per state per 800-point bank, labels included).
- Evidence: `papers/power_of_data_qml/results/photonic_scaling.md` (probe and
  production timings). The comparison mixes precision (float32 vs float64) and
  batch sizes, so it is indicative only.
- Why it matters: the GPU's value here was memory headroom (14 → 18 modes), not
  speed. If MerLin's GPU path is expected to be much faster, this may point to
  an under-used kernel.
- Suggested action: if a GPU benchmark exists, compare with these numbers;
  otherwise none.

### Low — Documentation: Fock-space size guidance for exact simulation

- Type: missing documentation
- What happened: with k = m/2 photons in m modes, the `FOCK` dimension
  C(m+k−1, k) controls memory. For 800 input points the state amplitudes
  took: m = 12, k = 6 (dimension 12,376) about 151 MB and under 1 s; m = 14,
  k = 7 (77,520) about 1 GB; m = 16 (about 490,000) was infeasible on a laptop.
  On an L4 with 45 GB of host RAM the ceiling was m = 18, k = 9 (3.1 M states),
  set by the 20 GB of stored states needed for the fidelity kernel. We found
  these limits by measurement.
- Evidence: `papers/power_of_data_qml/LOG.md`, "Phase 4 Photonic Assessment", cost
  paragraph; `LOG.md`, "Hardware-aware report", scaling limit. Timings and
  sizes are from one machine (Apple M5, MerLin 0.2.3, CPU, float64).
- Why it matters: users planning a photonic experiment cannot tell in advance
  which (m, k) fit on their machine.
- Suggested action: add a dimension and memory table (or a helper such as a
  `fock_dimension(m, k)` estimate with bytes per state) to the documentation.
