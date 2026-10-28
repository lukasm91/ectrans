GPU component round-trip tests

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
