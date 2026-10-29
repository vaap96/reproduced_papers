# Action Required: Send to MerLin Team

## Summary

Four findings from a photonic adaptation of Schuld, Sweke and Meyer (Fourier-series expressivity of data-encoding quantum models), all on MerLin 0.2.3 with Perceval 1.0.0:
- `add_angle_encoding` with several non-unit scales raises errors or silently encodes the wrong function;
- one `add_entangling_layer()` is not a universal mode unitary, contrary to the cookbook;
- `ComputationSpace.UNBUNCHED` is postselection with renormalisation, not threshold detection, and its outputs are not band-limited Fourier series;
- the cookbook targets 0.4.1 while the repository environment ships 0.2.3.

## Context

The paper shows that a data-encoding circuit is a Fourier series whose frequencies come from the encoding generator. The reproduction maps the generator to the photon number on the encoded modes and checks the predicted spectrum by FFT. This needs exact control over encodings, mesh universality and the meaning of the detection model, so the issues below affected the science directly. They are reproducible with the short scripts and tests named in each item; all evidence is in `papers/data_encoding_expressivity/`.

## Items to Send

### High — `CircuitBuilder.add_angle_encoding`: non-unit `scale` across several calls fails or silently mis-encodes

- Type: bug (probable; confirmed behaviour on 0.2.3, cause not investigated)
- What happened: several `add_angle_encoding` calls with different scales either raise an error in `forward`, or run and produce a spectrum different from the requested one.
  - Two calls (modes 1 and 2, scales 1 and 2, 4 modes, one photon): `IndexError: index 2 is out of bounds for dimension 1 with size 2`.
  - Five calls (modes 1 to 5, scales 1 to 5): runs silently, but the output spectrum has degree 1 instead of 5.
  - One call on `modes=[3]` with `scale=3.0` also raised `IndexError` in a first probe.
- Evidence: reproduced interactively on 0.2.3 (LOG.md, "Phase 4 Review", item 3). The working alternative is used in `lib/photonic_merlin.py`: one `add_angle_encoding(modes=..., scale=1.0)` call per layer, with inputs pre-scaled to `k_j·x`; `tests/test_photonic_merlin.py::test_multiscale_single_photon_spectrum_is_theorem_setting` checks the spectrum is exactly `{-5..5}`.
- Why it matters: a silently wrong spectrum is easy to miss, and per-feature scales are a common encoding choice.
- Suggested action: reproduce on 0.4.x; either support per-call scales or raise a clear error for conflicting scales.

### High — `add_entangling_layer`: one layer is not a universal mesh

- Type: documentation problem, with a possible improvement
- What happened: the cookbook states that "one trainable entangling layer is already a universal photonic mesh". In 0.2.3, one layer on m modes has m(m-1) trainable phases. A general mode unitary needs m² - 1 (modulo global phase).
- Evidence: Jacobian rank of the layer's unitary with respect to its parameters is 6, 12, 30 for m = 3, 4, 6 (needed: 8, 15, 35). A trainable phase screen (`add_rotations(modes=1..m-1, trainable=True)`) placed before the mesh gives rank 8, 15, 35. A screen placed after the mesh adds only one independent parameter (rank 7, 13, 31). Test: `tests/test_photonic_merlin.py::test_phase_screens_make_the_mesh_universal`.
- Why it matters: anything that relies on the mesh as a general unitary (universality arguments, expressivity studies) silently loses `m - 1` degrees of freedom. In the paper's setting, universal meshes were needed for the models to reach the target.
- Suggested action: correct the cookbook, or add a universal-mesh option (an input-side phase screen) to `add_entangling_layer`. Confirm on 0.4.x.

### Medium — `ComputationSpace.UNBUNCHED`: postselection and renormalisation, not threshold detection

- Type: documentation problem
- What happened: `UNBUNCHED` returns the collision-free outcomes of the true dynamics, restricted and renormalised. The cookbook describes it as "threshold detection". Physical threshold detectors keep bunched events as clicks. The renormalisation divides by an input-dependent probability, so the output is a ratio of trigonometric polynomials, not a band-limited Fourier series.
- Evidence: `tests/test_photonic_merlin.py::test_merlin_unbunched_is_postselected_and_renormalised_fock` (UNBUNCHED equals collision-free FOCK probabilities renormalised, to 1e-12; the discarded probability varies by more than 0.1 with the input). `::test_merlin_unbunched_output_is_not_band_limited` (for a two-photon circuit with a predicted degree of 2, the spectrum extends to the grid limit, with stray coefficients up to 0.13; `results/merlin_spectrum_summary.md`, last row). `UNBUNCHED` also rejects non-binary input states.
- Why it matters: Fourier-type or expectation-value analyses are valid for FOCK probabilities, or threshold click patterns derived from them, but not for the postselected output.
- Suggested action: document the postselection and renormalisation explicitly, and mention how to emulate threshold detection from `FOCK` (group outcomes by click pattern).

### Low — cookbook and environment version mismatch

- Type: documentation problem
- What happened: the cookbook states it was tested on 0.4.1 (`perceval-quandela >= 1.2.1`). The shared repository environment ships 0.2.3 with Perceval 1.0.0. Several cookbook APIs (for example `MeasurementStrategy` factory methods) differ in 0.2.3 (`MeasurementStrategy.PROBABILITIES`).
- Evidence: `pip show merlinquantum perceval-quandela` in the repository `.venv`. The Phase 4 code uses only the 0.2.3 API.
- Why it matters: reproductions copy cookbook snippets that fail on the pinned version. The findings above are on 0.2.3 and may not hold on 0.4.x; this has not been checked.
- Suggested action: pin or document the supported version in the repository requirements, and re-test the three items above on 0.4.x.
