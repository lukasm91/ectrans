Component tests

Backend-specific drivers and build rules live in cpu/ and gpu/.

CPU component tests
-------------------

The cpu/ drivers call the CPU stage interfaces in SP and DP. They need neither
GPU libraries nor a device. All four components use one/seven fields and
one/four MPI ranks when MPI is available:

1. grid_transpose: TRGTOL then TRLTOG on blocked grid arrays. Compare active
   points bitwise, including signed zero, excluding NPROMA padding. One wave
   set distributes fields across vertical sets, including empty local sets.

2. fourier_transpose: TRMTOL then TRLTOM on native CPU Fourier buffers. The
   two directions can have different peer-padding strides; bridge only those
   strides with local copies and compare the original/recovered active views
   bitwise. No MPI reference implementation or FFT is used by the comparison.

3. fft: FTDIR then FTINV on a reduced T23 Gaussian grid. Check normalized
   forward coefficients against independently synthesized Fourier modes and
   the inverse against the original real input. The CPU forward routine
   already normalizes by row length, so no additional inverse scaling is
   applied. Both checks use 64*epsilon(precision)*max(1,input scale), exclude
   padding and reject nonfinite results. Exercise per-field/batched FFT calls.

4. legendre: call the per-wave-number LTINV and LTDIR stages on deterministic
   scalar spectra, with imaginary m=0 coefficients zero. Full quadrature
   avoids reduced-grid polar truncation. Only local peer-padding copies join
   the two stages; no FFT or MPI transpose is included. Compare coefficients
   with 256*epsilon(precision)*max(1,input scale), rejecting nonfinite results.

The transpose and Legendre checks are round-trip identities, with the same
limitations as their GPU counterparts: they do not independently establish
one-way ordering or normalization. The FFT also has a forward reference check.

GPU component round-trip tests
-------------------------

Run from an OpenACC build with testing enabled:
  ctest -R '^ectrans_test_component_' --output-on-failure

The four tests compare original and recovered arrays:
- grid_transpose: TRGTOL/TRLTOG, bitwise on active NPROMA entries.
- fourier_transpose: TRLTOM/TRMTOL, bitwise; weighting is outside this boundary.
- fft: FTDIR/FTINV on band-limited real input, normalized by row length.
  The maximum error must be within 64 machine epsilons of the input scale.
- legendre: LTINV/LTDIR with Gaussian quadrature weights and real m=0 modes.
  The maximum error must be within 256 machine epsilons of the input scale.

The fixture uses truncation 23, 48 Gaussian latitudes and repeatable random
inputs. Tests cover both precisions, one and seven fields, and the project's
serial and MPI configurations. The grid test includes empty vertical sets.
Each executable also accepts the field count as its first argument.
