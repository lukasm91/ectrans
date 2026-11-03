! (C) Copyright 2025- ECMWF.
! (C) Copyright 2026- NVIDIA.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.

! Analytic verification: ECTRANS_TEST_TRANS_API_ANALYTIC_FIELDS
! -----------------------------------------------------------
! This case checks DIR_TRANS and INV_TRANS separately against analytic input
! and output, then checks linearity (TL) and both adjoints. It runs through the
! same public API on CPU/GPU, in SP/DP; references are evaluated in DP and cast
! to the tested kind. The older call-mode tests below remain separate cases.
!
! Fixture and four batched fields
!   The analytic case overrides the default T79 fixture with T63, 128 Gaussian
!   latitudes and reduced octahedral rows of 20+4*min(j,129-j) longitudes.
!   Each named field supplies a scalar s, streamfunction p=psi/R and velocity
!   potential c=chi/R, where R=PPEARTH_RADIUS. Their constructions are:
!     polynomial : (1+v.r)**8 for each channel, with unit Cartesian position r
!                  and the three fixed vectors in ZVECTORS; degree <= 8.
!     zonal      : degree/order (7,0), plus a constant scalar; no E-W variation.
!     short_scale: modes (24,0), (31,1), (46,3), including both degree parities,
!                  complex coefficients and m=1 winds near the poles.
!     weak_tail  : mode (4,1) plus a 1e-4-amplitude (47,3) mode, checked
!                  separately so the dominant mode cannot hide loss of the tail.
!   Wind-potential coefficients are scaled by 1/sqrt(n*(n+1)) in the harmonic
!   cases to keep wind amplitudes comparable across degrees. All retained
!   orders fit even the shortest polar rows; setup checks this explicitly.
!
! Analytic values, signs and derivatives
!   Let phi be latitude, lambda longitude, mu=sin(phi), and define angular
!   derivatives N=d/dphi and E=(1/cos(phi))*d/dlambda. Then
!     u = E(c)-N(p),  v = N(c)+E(p),
!     vorticity = Laplace_unit(p)/R, divergence = Laplace_unit(c)/R,
!   where Laplace_unit(Q_nm*exp(i*m*lambda))=-n*(n+1)*Q_nm*exp(i*m*lambda).
!   ANALYTIC_BASIS returns Q_nm and dQ_nm/dphi, not dQ_nm/dmu. It has Q_00=1,
!   no Condon--Shortley sign, and positive-m synthesis uses
!     2*Re(a_nm*Q_nm*exp(i*m*lambda)); m=0 is real and counted once.
!   POLYNOMIAL_VALUES differentiates the Cartesian polynomial explicitly;
!   POLYNOMIAL_COEFFICIENTS obtains its finite spectrum via the addition
!   theorem. HARMONIC_VALUES evaluates the other finite harmonic sums, with
!   an independent recurrence rather than production SUPOLF/SULEG tables.
!
!   DIR_TRANS receives analytic u, v, s and checks scalar/vorticity/divergence
!   coefficients. INV_TRANS starts from the exact coefficients, never from
!   DIR_TRANS output, and enables LDVORGP, LDDIVGP, LDSCDERS and LDUVDER.
!   Its nine output groups, each containing all four fields, are
!     vorticity, divergence, u, v, s, N(s)/R, E(u)/R, E(v)/R, E(s)/R.
!   E(u)/R and E(v)/R differentiate the local wind components in longitude;
!   they are not a complete covariant wind-gradient tensor. R factors are
!   removed for comparison, so dimensional derivatives do not look accurate
!   merely because their physical magnitudes are small. Only active grid
!   points are checked, excluding NPROMA padding.
!
! Checks and limits
!   Each field/output channel must be finite and have max absolute error <=
!   1024*epsilon(tested kind)*max(1,max(abs(reference))) after radius scaling.
!   The weak (47,3) coefficient additionally has a complex relative-error
!   bound of 2*epsilon(tested kind)/1e-4, separately for scalar/vor/div.
!   These finite, band-limited fields test derivatives and near-polar winds;
!   they do not test a discontinuity, an exact pole, or the T63 cutoff modes.
!
!   At fixed setup the transforms are linear, so ordinary DIR_TRANS/INV_TRANS
!   calls also implement TL. Check T(x+0.5*y)=T(x)+0.5*T(y), with y a cyclic
!   permutation of the four fields. For AD check <T*x,y>=<x,T_ad*y> for both
!   directions, with deterministic unequal dual weights. Grid dot products
!   are unweighted sums; spectral dot products count real m=0 once and both
!   real/imaginary positive-m parts twice. Dot products accumulate in DP;
!   normalized TL and AD errors use the same 1024*epsilon bound.
!   GPU AD requires stored Legendre coefficients; recurrence prints an
!   explicit skip. Other supported backends retain both adjoint checks.
!
! Running
!   Enable TESTS and the desired CPU/GPU and precision backends. CMake
!   registers serial, one-/two-rank MPI and one-/eight-thread variants where
!   available. To select just this case (rather than the whole API suite):
!     ctest --test-dir <build> -R '^ectrans_test_trans_api_analytic_fields_' \
!       --output-on-failure
!   One example test name is
!     ectrans_test_trans_api_analytic_fields_cpu_sp_mpi1xomp1
!

MODULE DIR_TRANS_TEST_SUITE

USE PARKIND1, ONLY: JPIM, JPRB, JPRD
USE MPL_MODULE, ONLY: MPL_INIT, MPL_NPROC, MPL_MYRANK, MPL_ALLREDUCE, MPL_END

IMPLICIT NONE

#include "setup_trans0.h"
#include "setup_trans.h"
#include "trans_inq.h"
#include "dist_grid.h"
#include "dir_trans.h"
#include "inv_trans.h"
#include "dir_transad.h"
#include "inv_transad.h"
#include "trans_end.h"

! Default spectral truncation; the analytic-fields case overrides this.
INTEGER(KIND=JPIM), PARAMETER :: JPTRUNCATION = 79

! Default number of latitudes; the analytic-fields case overrides this.
INTEGER(KIND=JPIM), PARAMETER :: JPNGDL = 2 * (JPTRUNCATION + 1)

! Tolerance for "close to zero"
REAL(KIND=JPRB), PARAMETER :: PPTOLERANCE = 100.0_JPRB * EPSILON(1.0_JPRB)

! JPPROMA blocking factor
INTEGER(KIND=JPIM), PARAMETER :: JPPROMA = 16

! Earth radius in metres
REAL(KIND=JPRD), PARAMETER :: PPEARTH_RADIUS = 6371229.0_JPRD

LOGICAL :: LUSE_MPI

CONTAINS

!---------------------------------------------------------------------------------------------------

! Approximate equality check for reals
ELEMENTAL LOGICAL FUNCTION APPROX_EQ(PA, PB, PTOL) RESULT(KRET)
  REAL(KIND=JPRB), INTENT(IN) :: PA
  REAL(KIND=JPRB), INTENT(IN) :: PB
  REAL(KIND=JPRB), INTENT(IN), OPTIONAL :: PTOL

  IF (PRESENT(PTOL)) THEN
    KRET = ABS(PA - PB) < PTOL
  ELSE
    KRET = ABS(PA - PB) < PPTOLERANCE
  END IF
END FUNCTION APPROX_EQ

! Initialise global field with all ones and distribute it
FUNCTION GET_INPUT_FIELD(KMY_PROC, KGPTOTG, KGPBLKS, KFIELDS) RESULT(PGP)
  INTEGER(KIND=JPIM), INTENT(IN) :: KMY_PROC
  INTEGER(KIND=JPIM), INTENT(IN) :: KGPTOTG
  INTEGER(KIND=JPIM), INTENT(IN) :: KGPBLKS
  INTEGER(KIND=JPIM), INTENT(IN) :: KFIELDS

  REAL(KIND=JPRB), ALLOCATABLE :: ZGPG(:,:), PGP(:,:,:)
  INTEGER(KIND=JPIM) :: I

  ! Initialise global field
  IF (KMY_PROC == 1) THEN
    ALLOCATE(ZGPG(KGPTOTG,KFIELDS))

    ! Set all of the input to one
    ZGPG(:,:) = 1.0_JPRB
  ENDIF

  ! Initialise distributed fields
  ALLOCATE(PGP(JPPROMA,KFIELDS,KGPBLKS))

  ! Distribute field from first task to other tasks
  IF (KMY_PROC == 1) THEN
    CALL DIST_GRID(PGPG=ZGPG, KFDISTG=KFIELDS, KFROM=[(1, I = 1, KFIELDS)], PGP=PGP, KPROMA=JPPROMA)
  ELSE
    CALL DIST_GRID(KFDISTG=KFIELDS, KFROM=[(1, I = 1, KFIELDS)], PGP=PGP, KPROMA=JPPROMA)
  ENDIF

  IF (KMY_PROC == 1) DEALLOCATE(ZGPG)
END FUNCTION GET_INPUT_FIELD

FUNCTION ROTATIONAL_WIND(KGPBLKS, K_REGIONS_NS, K_REGIONS_EW) RESULT(PGP)

  INTEGER(KIND=JPIM), INTENT(IN) :: KGPBLKS
  INTEGER(KIND=JPIM), INTENT(IN) :: K_REGIONS_NS
  INTEGER(KIND=JPIM), INTENT(IN) :: K_REGIONS_EW

  REAL(KIND=JPRB), ALLOCATABLE :: PGP(:,:,:)

  INTEGER(KIND=JPIM) :: IPTRFLOFF, IMY_REGION_NS, IMY_REGION_EW
  INTEGER(KIND=JPIM), DIMENSION(K_REGIONS_NS) :: IFRSTLAT, ILSTLAT
  REAL(KIND=JPRD) :: ZMU(JPNGDL)
  INTEGER(KIND=JPIM), DIMENSION(JPNGDL + K_REGIONS_NS - 1, K_REGIONS_EW) :: NSTA, NONL
  INTEGER(KIND=JPIM) :: ILAT, IFIRSTLAT, ILASTLAT, IBL, JROF, JGLAT, ISTLON, IENDLON, JLON
  REAL(KIND=JPRB) :: ZLAT

  CALL TRANS_INQ(KPTRFLOFF=IPTRFLOFF, KMY_REGION_NS=IMY_REGION_NS, KFRSTLAT=IFRSTLAT, &
    &            KLSTLAT=ILSTLAT, PMU=ZMU, KMY_REGION_EW=IMY_REGION_EW, KSTA=NSTA, KONL=NONL)
  ILAT = IPTRFLOFF
  IFIRSTLAT = IFRSTLAT(IMY_REGION_NS)
  ILASTLAT = ILSTLAT(IMY_REGION_NS)

  ALLOCATE(PGP(JPPROMA, 2, KGPBLKS))

  IBL = 1
  JROF = 1
  DO JGLAT = IFIRSTLAT, ILASTLAT
    ZLAT = ASIN(ZMU(JGLAT))
    ILAT = ILAT + 1
    ISTLON = NSTA(ILAT, IMY_REGION_EW)
    IENDLON = ISTLON - 1 + NONL(ILAT, IMY_REGION_EW)
    DO JLON = ISTLON, IENDLON
      PGP(JROF, 1, IBL) = COS(ZLAT) ! U = A * COS(THETA)
      PGP(JROF, 2, IBL) = 0.0_JPRB ! V = 0
      JROF = JROF + 1
      IF (JROF > JPPROMA) THEN
        JROF = 1
        IBL  = IBL + 1
      ENDIF
    ENDDO
  ENDDO

END FUNCTION ROTATIONAL_WIND

! Determine if this task handles the m=0 mode
LOGICAL FUNCTION HAVE_M0_MODE()
  INTEGER :: NUM_MY_ZON_WNS
  INTEGER, ALLOCATABLE :: MY_ZON_WNS(:)

  CALL TRANS_INQ(KNUMP=NUM_MY_ZON_WNS)
  ALLOCATE(MY_ZON_WNS(NUM_MY_ZON_WNS))
  CALL TRANS_INQ(KMYMS=MY_ZON_WNS)

  HAVE_M0_MODE = ANY(MY_ZON_WNS == 0)
END FUNCTION HAVE_M0_MODE

! Setup fixture
SUBROUTINE SETUP_TEST(KSPEC2, KGPTOTG, KGPTOT, KGPBLKS, KMY_PROC, K_REGIONS_NS, K_REGIONS_EW, KTRUNCATION)
  USE UTIL, ONLY: DETECT_MPIRUN

  INTEGER(KIND=JPIM), INTENT(OUT) :: KSPEC2
  INTEGER(KIND=JPIM), INTENT(OUT) :: KGPTOTG
  INTEGER(KIND=JPIM), INTENT(OUT) :: KGPTOT
  INTEGER(KIND=JPIM), INTENT(OUT) :: KGPBLKS
  INTEGER(KIND=JPIM), INTENT(OUT) :: KMY_PROC
  INTEGER(KIND=JPIM), INTENT(OUT) :: K_REGIONS_NS
  INTEGER(KIND=JPIM), INTENT(OUT) :: K_REGIONS_EW

  INTEGER(KIND=JPIM), OPTIONAL, INTENT(IN) :: KTRUNCATION
  INTEGER(KIND=JPIM), ALLOCATABLE :: ILOEN(:)
  INTEGER(KIND=JPIM) :: ITRUNCATION
  INTEGER(KIND=JPIM) :: I
  INTEGER(KIND=JPIM) :: IPROC

  ! Set up MPI
  LUSE_MPI = DETECT_MPIRUN()
  IF (LUSE_MPI) THEN
    CALL MPL_INIT
    IPROC = MPL_NPROC()
    KMY_PROC = MPL_MYRANK()
  ELSE
    IPROC = 1
    KMY_PROC = 1
  ENDIF

  CALL SETUP_TRANS0(LDMPOFF=.NOT. LUSE_MPI, KPRGPNS=IPROC, KPRGPEW=1, KPRTRW=IPROC, &
    &               K_REGIONS_NS=K_REGIONS_NS, K_REGIONS_EW=K_REGIONS_EW, PRAD=PPEARTH_RADIUS)

  ! Define octahedral grid
  ITRUNCATION = JPTRUNCATION
  IF (PRESENT(KTRUNCATION)) ITRUNCATION = KTRUNCATION
  ALLOCATE(ILOEN(2*(ITRUNCATION+1)))
  DO I = 1, ITRUNCATION + 1
    ILOEN(I) = 20 + 4 * I
    ILOEN(SIZE(ILOEN) - I + 1) = ILOEN(I)
  END DO

  CALL SETUP_TRANS(KSMAX=ITRUNCATION, KDGL=SIZE(ILOEN), KLOEN=ILOEN)

  CALL TRANS_INQ(KSPEC2=KSPEC2, KGPTOTG=KGPTOTG, KGPTOT=KGPTOT)

  ! Number of JPPROMA blocks
  KGPBLKS = (KGPTOT - 1) / JPPROMA + 1
END SUBROUTINE SETUP_TEST

! Tear down fixture
SUBROUTINE CLEANUP_TEST

  CALL TRANS_END

  IF (LUSE_MPI) THEN
    CALL MPL_END(LDMEMINFO=.FALSE.)
  ENDIF
END SUBROUTINE CLEANUP_TEST

!---------------------------------------------------------------------------------------------------

! The constant-field cases check that only the (0,0) scalar mode is nonzero,
! and that the wind fields have the expected vorticity and divergence.

!---------------------------------------------------------------------------------------------------

! Normalized spherical harmonics Q_nm(mu) for m<=8, without the Condon--Shortley
! sign, and latitude derivatives. This evaluator does not use SUPOLF/SULEG.
SUBROUTINE ANALYTIC_BASIS(PMU,KT,PQ,PDQ)
  INTEGER, INTENT(IN) :: KT
  REAL(JPRD), INTENT(IN) :: PMU
  REAL(JPRD), INTENT(OUT) :: PQ(0:KT,0:KT), PDQ(0:KT,0:KT)
  REAL(JPRD) :: ZC, ZA, ZB
  INTEGER :: N,M
  ZC = SQRT((1.0_JPRD-PMU)*(1.0_JPRD+PMU))
  PQ = 0.0_JPRD
  PDQ = 0.0_JPRD
  PQ(0,0) = 1.0_JPRD
  DO M=0,MIN(8,KT)
    IF (M > 0) THEN
      ZA = SQRT(REAL(2*M+1,JPRD)/REAL(2*M,JPRD))
      PQ(M,M) = ZA*ZC*PQ(M-1,M-1)
      PDQ(M,M) = ZA*(-PMU*PQ(M-1,M-1)+ZC*PDQ(M-1,M-1))
    ENDIF
    IF (M < KT) THEN
      ZA = SQRT(REAL(2*M+3,JPRD))
      PQ(M+1,M) = ZA*PMU*PQ(M,M)
      PDQ(M+1,M) = ZA*(ZC*PQ(M,M)+PMU*PDQ(M,M))
    ENDIF
    DO N=M+2,KT
      ZA = SQRT(REAL(4*N*N-1,JPRD)/REAL(N*N-M*M,JPRD))
      ZB = SQRT(REAL((2*N+1)*((N-1)**2-M*M),JPRD)/REAL((2*N-3)*(N*N-M*M),JPRD))
      PQ(N,M) = ZA*PMU*PQ(N-1,M)-ZB*PQ(N-2,M)
      PDQ(N,M) = ZA*(ZC*PQ(N-1,M)+PMU*PDQ(N-1,M))-ZB*PDQ(N-2,M)
    ENDDO
  ENDDO
END SUBROUTINE ANALYTIC_BASIS

! Exact coefficients of (1+v.r)**8 from its finite Legendre expansion and
! the addition theorem. Positive-m synthesis is 2*Re(a_nm*Q_nm*exp(i*m*lon)).
SUBROUTINE POLYNOMIAL_COEFFICIENTS(PVECTOR,PCOEFF)
  REAL(JPRD), INTENT(IN) :: PVECTOR(3)
  COMPLEX(JPRD), INTENT(OUT) :: PCOEFF(0:,0:)
  REAL(JPRD) :: ZR, ZLON, ZQ(0:8,0:8), ZDQ(0:8,0:8), ZSUM, ZTERM
  INTEGER :: N,M,K,J
  PCOEFF = CMPLX(0.0_JPRD,0.0_JPRD,JPRD)
  ZR = SQRT(SUM(PVECTOR**2))
  ZLON = ATAN2(PVECTOR(2),PVECTOR(1))
  CALL ANALYTIC_BASIS(PVECTOR(3)/ZR,8,ZQ,ZDQ)
  DO N=0,8
    ZSUM = 0.0_JPRD
    DO K=N,8,2
      ZTERM = ZR**K
      DO J=1,K
        ZTERM = ZTERM*REAL(9-J,JPRD)
      ENDDO
      DO J=K-N,1,-2
        ZTERM = ZTERM/REAL(J,JPRD)
      ENDDO
      DO J=K+N+1,1,-2
        ZTERM = ZTERM/REAL(J,JPRD)
      ENDDO
      ZSUM = ZSUM+ZTERM
    ENDDO
    DO M=0,N
      PCOEFF(N,M) = ZSUM*ZQ(N,M)*CMPLX(COS(M*ZLON),-SIN(M*ZLON),JPRD)
    ENDDO
  ENDDO
END SUBROUTINE POLYNOMIAL_COEFFICIENTS

! Value, R-scaled N-S/E-W gradient, R**2-scaled Laplacian, and longitude
! derivatives of the two R-scaled gradient components of (1+v.r)**8.
SUBROUTINE POLYNOMIAL_VALUES(PMU,PLON,PVECTOR,PVALUES)
  REAL(JPRD), INTENT(IN) :: PMU, PLON, PVECTOR(3)
  REAL(JPRD), INTENT(OUT) :: PVALUES(6)
  REAL(JPRD) :: ZC, ZS, ZFIRST, ZSECOND, ZA, ZB, ZNS
  ZC = SQRT((1.0_JPRD-PMU)*(1.0_JPRD+PMU))
  ZA = PVECTOR(1)*COS(PLON)+PVECTOR(2)*SIN(PLON)
  ZB = -PVECTOR(1)*SIN(PLON)+PVECTOR(2)*COS(PLON)
  ZS = ZC*ZA+PVECTOR(3)*PMU
  ZNS = -PMU*ZA+PVECTOR(3)*ZC
  ZFIRST = 8.0_JPRD*(1.0_JPRD+ZS)**7
  ZSECOND = 56.0_JPRD*(1.0_JPRD+ZS)**6
  PVALUES = [(1.0_JPRD+ZS)**8, ZFIRST*ZNS, ZFIRST*ZB, &
    & ZSECOND*(SUM(PVECTOR**2)-ZS**2)-2.0_JPRD*ZS*ZFIRST, &
    & ZSECOND*ZC*ZB*ZNS-ZFIRST*PMU*ZB, ZSECOND*ZC*ZB**2-ZFIRST*ZA]
END SUBROUTINE POLYNOMIAL_VALUES

SUBROUTINE HARMONIC_VALUES(PMU,PLON,PCOEFF,PQ,PDQ,PVALUES)
  REAL(JPRD), INTENT(IN) :: PMU, PLON, PQ(0:,0:), PDQ(0:,0:)
  COMPLEX(JPRD), INTENT(IN) :: PCOEFF(0:,0:,:)
  REAL(JPRD), INTENT(OUT) :: PVALUES(6,3)
  COMPLEX(JPRD) :: ZPHASE, ZI
  REAL(JPRD) :: ZC, ZRE(3), ZDLON(3), ZWEIGHT
  INTEGER :: N,M
  PVALUES = 0.0_JPRD
  ZC = SQRT((1.0_JPRD-PMU)*(1.0_JPRD+PMU))
  ZI = CMPLX(0.0_JPRD,1.0_JPRD,JPRD)
  DO M=0,3
    ZPHASE = CMPLX(COS(M*PLON),SIN(M*PLON),JPRD)
    ZWEIGHT = REAL(MERGE(1,2,M==0),JPRD)
    DO N=M,47
      IF (ALL(PCOEFF(N,M,:) == CMPLX(0.0_JPRD,0.0_JPRD,JPRD))) CYCLE
      ZRE = ZWEIGHT*REAL(PCOEFF(N,M,:)*ZPHASE,JPRD)
      ZDLON = ZWEIGHT*REAL(ZI*REAL(M,JPRD)*PCOEFF(N,M,:)*ZPHASE,JPRD)
      PVALUES(1,:) = PVALUES(1,:)+PQ(N,M)*ZRE
      PVALUES(2,:) = PVALUES(2,:)+PDQ(N,M)*ZRE
      PVALUES(3,:) = PVALUES(3,:)+PQ(N,M)*ZDLON/ZC
      PVALUES(4,:) = PVALUES(4,:)-REAL(N*(N+1),JPRD)*PQ(N,M)*ZRE
      PVALUES(5,:) = PVALUES(5,:)+PDQ(N,M)*ZDLON
      PVALUES(6,:) = PVALUES(6,:)-REAL(M*M,JPRD)*PQ(N,M)*ZRE/ZC
    ENDDO
  ENDDO
END SUBROUTINE HARMONIC_VALUES

! One reduced-grid test, four batched scalar/streamfunction/velocity-potential
! constructions. Direct and inverse use independent inputs, not a round trip.
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_ANALYTIC_FIELDS() RESULT(KRET) BIND(C)
  USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
  INTEGER, PARAMETER :: IT=63, INLAT=2*(IT+1), IFIELDS=4
  REAL(JPRD), PARAMETER :: ZWEAK=1.0E-4_JPRD
  REAL(JPRD), PARAMETER :: ZVECTORS(3,3)=RESHAPE([0.3_JPRD,-0.4_JPRD,0.2_JPRD, &
    & -0.2_JPRD,0.1_JPRD,0.45_JPRD,0.3_JPRD,-0.4_JPRD,0.2_JPRD],[3,3])
  CHARACTER(LEN=16), PARAMETER :: CNAMES(IFIELDS) = &
    & [CHARACTER(LEN=16) :: 'polynomial','zonal','short_scale','weak_tail']
  COMPLEX(JPRD) :: ZCOEFF(0:IT,0:IT,3,IFIELDS), ZEXPECTED(3)
  REAL(JPRD) :: ZMU(INLAT), ZQ(0:IT,0:IT), ZDQ(0:IT,0:IT), ZVALUES(6,3), ZGRID(9)
  REAL(JPRD) :: ZLON, ZC, ZPI, ZFACTOR, ZVALUE, ZTOL, ZWEAKTOL
  REAL(JPRD) :: ZGRID_ERROR(IFIELDS,9), ZGRID_SCALE(IFIELDS,9)
  REAL(JPRD) :: ZSPEC_ERROR(IFIELDS,3), ZSPEC_SCALE(IFIELDS,3), ZWEAK_ERROR(3)
  REAL(JPRB), ALLOCATABLE :: ZGLOBAL(:,:), ZREFERENCE(:,:,:), ZINPUT(:,:,:), ZOUTPUT(:,:,:)
  REAL(JPRB), ALLOCATABLE :: ZEXACT(:,:,:), ZGOT(:,:,:)
  INTEGER :: ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW, INUMP
  INTEGER :: IMEN(INLAT), IASM0(0:IT)
  INTEGER, ALLOCATABLE :: IMYMS(:)
  INTEGER :: JLAT, JLON, ILOEN, IPOINT, JBLK, JROW, JFIELD, J, N, M, INDEX, JM, JPART

  CALL SETUP_TEST(ISPEC2,IGPTOTG,IGPTOT,IGPBLKS,IMY_PROC,I_REGIONS_NS,I_REGIONS_EW,IT)
  CALL TRANS_INQ(PMU=ZMU,KNMENG=IMEN,KNUMP=INUMP,KASM0=IASM0)
  ALLOCATE(IMYMS(INUMP))
  CALL TRANS_INQ(KMYMS=IMYMS)
  ! All fields have m<=8, including the polynomial. Avoid filtering the oracle.
  IF (MINVAL(IMEN) < 8) THEN
    KRET=1
    CALL CLEANUP_TEST
    RETURN
  ENDIF

  ZCOEFF = CMPLX(0.0_JPRD,0.0_JPRD,JPRD)
  DO J=1,3
    CALL POLYNOMIAL_COEFFICIENTS(ZVECTORS(:,J),ZCOEFF(:,:,J,1))
  ENDDO
  ! The other fields have explicit normalized harmonic coefficients. The last
  ! two channels are psi/R and chi/R; divide their coefficients by sqrt(n(n+1))
  ! to keep wind amplitudes comparable between degrees. Zonal derivatives in
  ! longitude must vanish. The short-scale case exercises both parities and
  ! nonzero m=1 polar wind; the weak-tail case isolates a 1e-4 high-degree mode.
  ZCOEFF(0,0,1,2)=0.2_JPRD
  ZCOEFF(7,0,:,2)=[0.7_JPRD,1.0_JPRD/SQRT(56.0_JPRD),-0.4_JPRD/SQRT(56.0_JPRD)]
  ZCOEFF(24,0,:,3)=[0.15_JPRD,-0.1_JPRD/SQRT(600.0_JPRD),0.2_JPRD/SQRT(600.0_JPRD)]
  ZCOEFF(31,1,:,3)=[CMPLX(0.2_JPRD,-0.15_JPRD,JPRD), &
    & CMPLX(0.4_JPRD,0.1_JPRD,JPRD)/SQRT(992.0_JPRD), &
    & CMPLX(-0.2_JPRD,0.3_JPRD,JPRD)/SQRT(992.0_JPRD)]
  ZCOEFF(46,3,:,3)=[CMPLX(0.07_JPRD,0.09_JPRD,JPRD), &
    & CMPLX(0.3_JPRD,-0.2_JPRD,JPRD)/SQRT(2162.0_JPRD), &
    & CMPLX(0.1_JPRD,0.25_JPRD,JPRD)/SQRT(2162.0_JPRD)]
  ZCOEFF(4,1,:,4)=[CMPLX(0.3_JPRD,0.2_JPRD,JPRD), &
    & CMPLX(0.2_JPRD,-0.1_JPRD,JPRD)/SQRT(20.0_JPRD), &
    & CMPLX(-0.1_JPRD,0.3_JPRD,JPRD)/SQRT(20.0_JPRD)]
  ZCOEFF(47,3,:,4)=ZWEAK*[CMPLX(0.6_JPRD,-0.8_JPRD,JPRD), &
    & CMPLX(0.8_JPRD,0.6_JPRD,JPRD)/SQRT(2256.0_JPRD), &
    & CMPLX(-0.6_JPRD,0.8_JPRD,JPRD)/SQRT(2256.0_JPRD)]

  ALLOCATE(ZREFERENCE(JPPROMA,9*IFIELDS,IGPBLKS),ZINPUT(JPPROMA,3*IFIELDS,IGPBLKS))
  ALLOCATE(ZOUTPUT(JPPROMA,9*IFIELDS,IGPBLKS),ZEXACT(IFIELDS,ISPEC2,3),ZGOT(IFIELDS,ISPEC2,3))
  ZREFERENCE=0.0_JPRB
  ZPI=ACOS(-1.0_JPRD)
  IF (IMY_PROC == 1) THEN
    ALLOCATE(ZGLOBAL(IGPTOTG,9*IFIELDS))
    IPOINT=0
    DO JLAT=1,INLAT
      ILOEN=20+4*MIN(JLAT,INLAT+1-JLAT)
      ZC=SQRT((1.0_JPRD-ZMU(JLAT))*(1.0_JPRD+ZMU(JLAT)))
      CALL ANALYTIC_BASIS(ZMU(JLAT),IT,ZQ,ZDQ)
      DO JLON=1,ILOEN
        ZLON=2.0_JPRD*ZPI*REAL(JLON-1,JPRD)/REAL(ILOEN,JPRD)
        IPOINT=IPOINT+1
        DO JFIELD=1,IFIELDS
          IF (JFIELD == 1) THEN
            DO J=1,3
              CALL POLYNOMIAL_VALUES(ZMU(JLAT),ZLON,ZVECTORS(:,J),ZVALUES(:,J))
            ENDDO
          ELSE
            CALL HARMONIC_VALUES(ZMU(JLAT),ZLON,ZCOEFF(:,:,:,JFIELD),ZQ,ZDQ,ZVALUES)
          ENDIF
          ! API output: vor,div,u,v,scalar,scalar_NS,u_EW,v_EW,scalar_EW.
          ! Derivatives and vor/div are R-scaled to avoid tiny dimensional norms.
          ZGRID=[ZVALUES(4,2),ZVALUES(4,3),ZVALUES(3,3)-ZVALUES(2,2), &
            & ZVALUES(2,3)+ZVALUES(3,2),ZVALUES(1,1),ZVALUES(2,1), &
            & (ZVALUES(6,3)-ZVALUES(5,2))/ZC,(ZVALUES(5,3)+ZVALUES(6,2))/ZC,ZVALUES(3,1)]
          DO J=1,9
            ZGLOBAL(IPOINT,(J-1)*IFIELDS+JFIELD)=REAL(ZGRID(J),JPRB)
          ENDDO
        ENDDO
      ENDDO
    ENDDO
    CALL DIST_GRID(PGPG=ZGLOBAL,KFDISTG=9*IFIELDS,KFROM=[(1,J=1,9*IFIELDS)], &
      & PGP=ZREFERENCE,KPROMA=JPPROMA)
    DEALLOCATE(ZGLOBAL)
  ELSE
    CALL DIST_GRID(KFDISTG=9*IFIELDS,KFROM=[(1,J=1,9*IFIELDS)],PGP=ZREFERENCE,KPROMA=JPPROMA)
  ENDIF

  ZEXACT=0.0_JPRB
  DO JM=1,INUMP
    M=IMYMS(JM)
    DO N=M,IT
      INDEX=IASM0(M)+2*(N-M)
      DO JFIELD=1,IFIELDS
        ZEXPECTED=[ZCOEFF(N,M,1,JFIELD), &
          & -REAL(N*(N+1),JPRD)*ZCOEFF(N,M,2,JFIELD)/PPEARTH_RADIUS, &
          & -REAL(N*(N+1),JPRD)*ZCOEFF(N,M,3,JFIELD)/PPEARTH_RADIUS]
        ZEXACT(JFIELD,INDEX,:)=REAL(ZEXPECTED,JPRB)
        ZEXACT(JFIELD,INDEX+1,:)=REAL(AIMAG(ZEXPECTED),JPRB)
      ENDDO
    ENDDO
  ENDDO
  ZINPUT(:,1:2*IFIELDS,:)=ZREFERENCE(:,2*IFIELDS+1:4*IFIELDS,:)
  ZINPUT(:,2*IFIELDS+1:3*IFIELDS,:)=ZREFERENCE(:,4*IFIELDS+1:5*IFIELDS,:)
  CALL DIR_TRANS(PGP=ZINPUT,PSPSCALAR=ZGOT(:,:,1),PSPVOR=ZGOT(:,:,2),PSPDIV=ZGOT(:,:,3),KPROMA=JPPROMA)
  ! Independently initialize inverse input; never use the direct result here.
  CALL INV_TRANS(PSPSCALAR=ZEXACT(:,:,1),PSPVOR=ZEXACT(:,:,2),PSPDIV=ZEXACT(:,:,3),PGP=ZOUTPUT, &
    & LDVORGP=.TRUE.,LDDIVGP=.TRUE.,LDSCDERS=.TRUE.,LDUVDER=.TRUE.,KPROMA=JPPROMA)

  KRET=0
  ZGRID_ERROR=0.0_JPRD
  ZGRID_SCALE=1.0_JPRD
  ZSPEC_ERROR=0.0_JPRD
  ZSPEC_SCALE=1.0_JPRD
  ZWEAK_ERROR=0.0_JPRD
  DO J=1,9
    DO JFIELD=1,IFIELDS
      DO IPOINT=1,IGPTOT
        JBLK=(IPOINT-1)/JPPROMA+1
        JROW=MOD(IPOINT-1,JPPROMA)+1
        ZVALUE=REAL(ZOUTPUT(JROW,(J-1)*IFIELDS+JFIELD,JBLK),JPRD)
        IF (.NOT. IEEE_IS_FINITE(ZVALUE)) KRET=1
        IF (J<=2 .OR. J>=6) ZVALUE=ZVALUE*PPEARTH_RADIUS
        ZGRID_ERROR(JFIELD,J)=MAX(ZGRID_ERROR(JFIELD,J), &
          & ABS(ZVALUE-REAL(ZREFERENCE(JROW,(J-1)*IFIELDS+JFIELD,JBLK),JPRD)))
        ZGRID_SCALE(JFIELD,J)=MAX(ZGRID_SCALE(JFIELD,J), &
          & ABS(REAL(ZREFERENCE(JROW,(J-1)*IFIELDS+JFIELD,JBLK),JPRD)))
      ENDDO
    ENDDO
  ENDDO
  DO JPART=1,3
    ZFACTOR=1.0_JPRD
    IF (JPART>1) ZFACTOR=PPEARTH_RADIUS
    DO JFIELD=1,IFIELDS
      IF (.NOT. ALL(IEEE_IS_FINITE(ZGOT(JFIELD,:,JPART)))) KRET=1
      IF (ISPEC2==0) CYCLE
      ZSPEC_ERROR(JFIELD,JPART)=ZFACTOR*MAXVAL(ABS(REAL(ZGOT(JFIELD,:,JPART),JPRD) &
        & -REAL(ZEXACT(JFIELD,:,JPART),JPRD)))
      ZSPEC_SCALE(JFIELD,JPART)=MAX(1.0_JPRD,ZFACTOR*MAXVAL(ABS(REAL(ZEXACT(JFIELD,:,JPART),JPRD))))
    ENDDO
    IF (ANY(IMYMS==3)) THEN
      INDEX=IASM0(3)+2*(47-3)
      ZWEAK_ERROR(JPART)=SQRT(SUM((REAL(ZGOT(4,INDEX:INDEX+1,JPART),JPRD) &
        & -REAL(ZEXACT(4,INDEX:INDEX+1,JPART),JPRD))**2)) &
        & /SQRT(SUM(REAL(ZEXACT(4,INDEX:INDEX+1,JPART),JPRD)**2))
    ENDIF
  ENDDO
  IF (LUSE_MPI) THEN
    CALL MPL_ALLREDUCE(ZGRID_ERROR,CDOPER='MAX')
    CALL MPL_ALLREDUCE(ZGRID_SCALE,CDOPER='MAX')
    CALL MPL_ALLREDUCE(ZSPEC_ERROR,CDOPER='MAX')
    CALL MPL_ALLREDUCE(ZSPEC_SCALE,CDOPER='MAX')
    CALL MPL_ALLREDUCE(ZWEAK_ERROR,CDOPER='MAX')
    CALL MPL_ALLREDUCE(KRET,CDOPER='MAX')
  ENDIF
  ZTOL=1024.0_JPRD*REAL(EPSILON(1.0_JPRB),JPRD)
  ! Resolve the weak coefficient separately: allow rounding amplified by its
  ! 1e-4 amplitude, rather than applying the much larger full-field tolerance.
  ZWEAKTOL=2.0_JPRD*REAL(EPSILON(1.0_JPRB),JPRD)/ZWEAK
  IF (ANY(ZGRID_ERROR>ZTOL*ZGRID_SCALE) .OR. ANY(ZSPEC_ERROR>ZTOL*ZSPEC_SCALE)) KRET=1
  IF (ANY(ZWEAK_ERROR>ZWEAKTOL)) KRET=1
  IF (IMY_PROC==1) THEN
    DO JFIELD=1,IFIELDS
      WRITE(*,*) 'Analytic field: ',TRIM(CNAMES(JFIELD))
      WRITE(*,*) 'Direct scalar/vor/div scaled max errors:',ZSPEC_ERROR(JFIELD,:)/ZSPEC_SCALE(JFIELD,:)
      WRITE(*,*) 'Inverse vor/div/u/v/scalar/scalar_NS/u_EW/v_EW/scalar_EW errors:', &
        & ZGRID_ERROR(JFIELD,:)/ZGRID_SCALE(JFIELD,:)
    ENDDO
    WRITE(*,*) 'Weak-mode scalar/vor/div relative errors:',ZWEAK_ERROR
    WRITE(*,*) 'Global and weak-mode tolerances:',ZTOL,ZWEAKTOL
  ENDIF
  CALL CHECK_ANALYTIC_TL_AD(IMY_PROC,IGPTOT,IMYMS,IASM0,ZINPUT,ZEXACT,ZGOT,ZOUTPUT, &
    & ZREFERENCE,ZGRID_SCALE,ZSPEC_SCALE,ZTOL,KRET)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_ANALYTIC_FIELDS

!---------------------------------------------------------------------------------------------------

! ecTrans is linear at fixed resolution/configuration, so its ordinary calls
! are also the TL operator. Test superposition with a cyclic field perturbation.
! AD uses an unweighted grid dot product and spectral weights 1 (real m=0)
! and 2 (real/imag m>0), matching the library's adjoint convention.
SUBROUTINE CHECK_ANALYTIC_TL_AD(KMY_PROC,KGPTOT,KMYMS,KASM0,PGPIN,PSPIN,PSPOUT,PGPOUT, &
  & PREF,PGSCALE,PSSCALE,PTOL,KRET)
  USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_IS_FINITE
  INTEGER, INTENT(IN) :: KMY_PROC,KGPTOT,KMYMS(:),KASM0(0:)
  REAL(JPRB), INTENT(IN) :: PGPIN(:,:,:),PSPIN(:,:,:),PSPOUT(:,:,:),PGPOUT(:,:,:),PREF(:,:,:)
  REAL(JPRD), INTENT(IN) :: PGSCALE(:,:),PSSCALE(:,:),PTOL
  INTEGER, INTENT(INOUT) :: KRET
  REAL(JPRB), ALLOCATABLE :: ZGIN(:,:,:),ZGOUT(:,:,:),ZSPEC(:,:,:)
  REAL(JPRD) :: ZTLERROR(SIZE(PSPIN,1),2), ZDOT(SIZE(PSPIN,1),2,2), ZADERROR(SIZE(PSPIN,1),2)
  REAL(JPRD) :: ZVALUE,ZSCALE,ZFACTOR
  INTEGER :: KF,F,FNEXT,J,P,B,R,I,IGROUP
  CHARACTER(LEN=32) :: COPTION
  LOGICAL :: LAD

  KF=SIZE(PSPIN,1)
  ALLOCATE(ZGIN(SIZE(PGPIN,1),SIZE(PGPIN,2),SIZE(PGPIN,3)))
  ALLOCATE(ZGOUT(SIZE(PGPOUT,1),SIZE(PGPOUT,2),SIZE(PGPOUT,3)))
  ALLOCATE(ZSPEC(SIZE(PSPIN,1),SIZE(PSPIN,2),SIZE(PSPIN,3)))
  ZTLERROR=0.0_JPRD
  DO J=1,3
    I=(J-1)*KF
    ZGIN(:,I+1:I+KF,:)=PGPIN(:,I+1:I+KF,:)+0.5_JPRB*CSHIFT(PGPIN(:,I+1:I+KF,:),1,DIM=2)
  ENDDO
  CALL DIR_TRANS(PGP=ZGIN,PSPSCALAR=ZSPEC(:,:,1),PSPVOR=ZSPEC(:,:,2),PSPDIV=ZSPEC(:,:,3),KPROMA=JPPROMA)
  IF (.NOT. ALL(IEEE_IS_FINITE(ZSPEC))) KRET=1
  DO J=1,3
    ZFACTOR=1.0_JPRD
    IF (J>1) ZFACTOR=PPEARTH_RADIUS
    DO F=1,KF
      FNEXT=MOD(F,KF)+1
      ZSCALE=PSSCALE(F,J)+0.5_JPRD*PSSCALE(FNEXT,J)
      IF (SIZE(ZSPEC,2)==0) CYCLE
      ZVALUE=ZFACTOR*MAXVAL(ABS(REAL(ZSPEC(F,:,J),JPRD)-REAL(PSPOUT(F,:,J),JPRD) &
        & -0.5_JPRD*REAL(PSPOUT(FNEXT,:,J),JPRD)))/ZSCALE
      ZTLERROR(F,1)=MAX(ZTLERROR(F,1),ZVALUE)
    ENDDO
  ENDDO
  ZSPEC=PSPIN+0.5_JPRB*CSHIFT(PSPIN,1,DIM=1)
  CALL INV_TRANS(PSPSCALAR=ZSPEC(:,:,1),PSPVOR=ZSPEC(:,:,2),PSPDIV=ZSPEC(:,:,3),PGP=ZGOUT, &
    & LDVORGP=.TRUE.,LDDIVGP=.TRUE.,LDSCDERS=.TRUE.,LDUVDER=.TRUE.,KPROMA=JPPROMA)
  DO J=1,9
    ZFACTOR=1.0_JPRD
    IF (J<=2 .OR. J>=6) ZFACTOR=PPEARTH_RADIUS
    DO F=1,KF
      FNEXT=MOD(F,KF)+1
      ZSCALE=PGSCALE(F,J)+0.5_JPRD*PGSCALE(FNEXT,J)
      DO P=1,KGPTOT
        B=(P-1)/JPPROMA+1
        R=MOD(P-1,JPPROMA)+1
        ZVALUE=REAL(ZGOUT(R,(J-1)*KF+F,B),JPRD)
        IF (.NOT. IEEE_IS_FINITE(ZVALUE)) KRET=1
        ZVALUE=ZFACTOR*ABS(ZVALUE-REAL(PGPOUT(R,(J-1)*KF+F,B),JPRD) &
          & -0.5_JPRD*REAL(PGPOUT(R,(J-1)*KF+FNEXT,B),JPRD))/ZSCALE
        ZTLERROR(F,2)=MAX(ZTLERROR(F,2),ZVALUE)
      ENDDO
    ENDDO
  ENDDO
  IF (LUSE_MPI) CALL MPL_ALLREDUCE(ZTLERROR,CDOPER='MAX')
  IF (ANY(ZTLERROR>PTOL)) KRET=1
  IF (KMY_PROC==1) WRITE(*,*) 'TL superposition errors per field (direct, inverse):',ZTLERROR

  LAD=.TRUE.
#ifdef ECTRANS_TEST_GPU
  COPTION=''
  CALL GET_ENVIRONMENT_VARIABLE('ECTRANS_LEGENDRE_STORAGE',COPTION)
  IF (TRIM(COPTION)=='recurrence') LAD=.FALSE.
#endif
  IF (LAD) THEN
    ! Deterministic duals reuse the four fields with unequal positive weights.
    ! R**2 balances vorticity/divergence against scalar coefficients in D's dot
    ! product; R balances dimensional derivative outputs in I's dot product.
    DO J=1,3
      ZFACTOR=1.0_JPRD
      IF (J>1) ZFACTOR=PPEARTH_RADIUS**2
      DO F=1,KF
        DO I=1,SIZE(ZSPEC,2)
          ZSPEC(F,I,J)=REAL(REAL(PSPIN(F,I,J),JPRD)*ZFACTOR/PSSCALE(F,J)**2 &
            & *(1.0_JPRD+0.25_JPRD*SIN(REAL(I+13*F+J,JPRD))),JPRB)
        ENDDO
      ENDDO
    ENDDO
    ZDOT(:,1,1)=ANALYTIC_SPEC_DOT(PSPOUT,ZSPEC,KMYMS,KASM0)
    ! DIR_TRANSAD may consume its spectral inputs; form the dot product first.
    CALL DIR_TRANSAD(PSPSCALAR=ZSPEC(:,:,1),PSPVOR=ZSPEC(:,:,2),PSPDIV=ZSPEC(:,:,3), &
      & PGP=ZGIN,KPROMA=JPPROMA)
    ZDOT(:,2,1)=ANALYTIC_GRID_DOT(PGPIN,ZGIN,KGPTOT,KF)
    DO J=1,9
      ZFACTOR=1.0_JPRD
      IF (J<=2 .OR. J>=6) ZFACTOR=PPEARTH_RADIUS
      DO F=1,KF
        IGROUP=(J-1)*KF+F
        ZGOUT(:,IGROUP,:)=REAL(REAL(PREF(:,IGROUP,:),JPRD)*ZFACTOR/PGSCALE(F,J)**2,JPRB)
        DO P=1,KGPTOT
          B=(P-1)/JPPROMA+1
          R=MOD(P-1,JPPROMA)+1
          ZGOUT(R,IGROUP,B)=ZGOUT(R,IGROUP,B) &
            & *REAL(1.0_JPRD+0.25_JPRD*SIN(REAL(P+17*F+3*J,JPRD)),JPRB)
        ENDDO
      ENDDO
    ENDDO
    ZDOT(:,1,2)=ANALYTIC_GRID_DOT(PGPOUT,ZGOUT,KGPTOT,KF)
    ! The current AD implementation accumulates into spectral output buffers.
    ZSPEC=0.0_JPRB
    CALL INV_TRANSAD(PSPSCALAR=ZSPEC(:,:,1),PSPVOR=ZSPEC(:,:,2),PSPDIV=ZSPEC(:,:,3),PGP=ZGOUT, &
      & LDVORGP=.TRUE.,LDDIVGP=.TRUE.,LDSCDERS=.TRUE.,LDUVDER=.TRUE.,KPROMA=JPPROMA)
    IF (.NOT. ALL(IEEE_IS_FINITE(ZSPEC))) KRET=1
    ZDOT(:,2,2)=ANALYTIC_SPEC_DOT(PSPIN,ZSPEC,KMYMS,KASM0)
    IF (LUSE_MPI) THEN
      DO J=1,2
        CALL MPL_ALLREDUCE(ZDOT(:,:,J),CDOPER='SUM',LDREPROD=.FALSE.)
      ENDDO
    ENDIF
    IF (.NOT. ALL(IEEE_IS_FINITE(ZDOT))) KRET=1
    ZADERROR=ABS(ZDOT(:,1,:)-ZDOT(:,2,:)) &
      & /MAX(ABS(ZDOT(:,1,:)),ABS(ZDOT(:,2,:)),TINY(1.0_JPRD))
    IF (ANY(ZADERROR>PTOL)) KRET=1
    IF (KMY_PROC==1) WRITE(*,*) 'AD dot-product errors per field (direct, inverse):',ZADERROR
  ELSEIF (KMY_PROC==1) THEN
    WRITE(*,*) 'AD unavailable for this GPU configuration: requires stored coefficients.'
  ENDIF
  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET,CDOPER='MAX')
END SUBROUTINE CHECK_ANALYTIC_TL_AD

FUNCTION ANALYTIC_SPEC_DOT(PA,PB,KMYMS,KASM0) RESULT(PDOT)
  REAL(JPRB), INTENT(IN) :: PA(:,:,:),PB(:,:,:)
  INTEGER, INTENT(IN) :: KMYMS(:),KASM0(0:)
  REAL(JPRD) :: PDOT(SIZE(PA,1)),ZWEIGHT
  INTEGER :: JM,M,N,JRI,I,J
  PDOT=0.0_JPRD
  DO JM=1,SIZE(KMYMS)
    M=KMYMS(JM)
    ZWEIGHT=REAL(MERGE(1,2,M==0),JPRD)
    DO N=M,UBOUND(KASM0,1)
      DO JRI=0,MIN(1,M)
        I=KASM0(M)+2*(N-M)+JRI
        DO J=1,SIZE(PA,3)
          PDOT=PDOT+ZWEIGHT*REAL(PA(:,I,J),JPRD)*REAL(PB(:,I,J),JPRD)
        ENDDO
      ENDDO
    ENDDO
  ENDDO
END FUNCTION ANALYTIC_SPEC_DOT

FUNCTION ANALYTIC_GRID_DOT(PA,PB,KGPTOT,KF) RESULT(PDOT)
  REAL(JPRB), INTENT(IN) :: PA(:,:,:),PB(:,:,:)
  INTEGER, INTENT(IN) :: KGPTOT,KF
  REAL(JPRD) :: PDOT(KF)
  INTEGER :: P,B,R,F,J
  PDOT=0.0_JPRD
  DO P=1,KGPTOT
    B=(P-1)/JPPROMA+1
    R=MOD(P-1,JPPROMA)+1
    DO J=1,SIZE(PA,2)
      F=MOD(J-1,KF)+1
      PDOT(F)=PDOT(F)+REAL(PA(R,J,B),JPRD)*REAL(PB(R,J,B),JPRD)
    ENDDO
  ENDDO
END FUNCTION ANALYTIC_GRID_DOT

! Test DIR_TRANS with call mode 1 and just one scalar field
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_SCALAR_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGP(:,:,:), ZSPSCALAR(:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ZGP = GET_INPUT_FIELD(IMY_PROC, IGPTOTG, IGPBLKS, 1)
  ALLOCATE(ZSPSCALAR(1,ISPEC2))

  CALL DIR_TRANS(PGP=ZGP, PSPSCALAR=ZSPSCALAR, KPROMA=JPPROMA)

  ! Check only the (0,0) mode (global mean) is one and all the rest are zero
  IF (HAVE_M0_MODE()) THEN
    KRET = MERGE(0, 1, APPROX_EQ(ZSPSCALAR(1,1), 1.0_JPRB) .AND. &
      &         ALL(APPROX_EQ(ZSPSCALAR(1,2:), 0.0_JPRB)))
  ELSE
    KRET = MERGE(0, 1, ALL(APPROX_EQ(ZSPSCALAR, 0.0_JPRB)))
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPSCALAR, ZGP)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_SCALAR_1

!---------------------------------------------------------------------------------------------------

! Test DIR_TRANS with call mode 1 and 1-level wind fields
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_WIND_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGP(:,:,:), ZSPVOR(:,:), ZSPDIV(:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW
  REAL(KIND=JPRB) :: ZCORRECT_VORTICITY_VALUE

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ZGP = ROTATIONAL_WIND(IGPBLKS, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZSPVOR(1,ISPEC2))
  ALLOCATE(ZSPDIV(1,ISPEC2))

  CALL DIR_TRANS(PGP=ZGP, PSPVOR=ZSPVOR, PSPDIV=ZSPDIV, KPROMA=JPPROMA)

  IF (HAVE_M0_MODE()) THEN
    ! epsilon_{m=0}^{n=1} = sqrt((n^2 - m^2)/(4n^2 - 1)) = sqrt(1/3)
    ZCORRECT_VORTICITY_VALUE = 2.0_JPRB * SQRT(1.0_JPRB / 3.0_JPRB) / PPEARTH_RADIUS

    ! Check all criteria
    KRET = 0

    ! Check divergence is all zero
    IF (ANY(.NOT. APPROX_EQ(ZSPDIV, 0.0_JPRB))) KRET = 1

     ! Check all irrelevant vorticity modes are zero
    IF (ANY(.NOT. APPROX_EQ(ZSPVOR(1,1:2), 0.0_JPRB))) KRET = 1

    ! Check selected harmonic has the correct value
    IF (.NOT. APPROX_EQ(ZSPVOR(1,3), ZCORRECT_VORTICITY_VALUE)) KRET = 1
  ELSE
    KRET = MERGE(0, 1, ALL(APPROX_EQ(ZSPVOR, 0.0_JPRB) .AND. ALL(APPROX_EQ(ZSPDIV, 0.0_JPRB))))
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPDIV, ZSPVOR, ZGP)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_WIND_1

!---------------------------------------------------------------------------------------------------

! Test DIR_TRANS with call mode 1 and 1-level wind fields and 1 scalar field
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_WIND_1_SCALAR_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGP(:,:,:), ZSPVOR(:,:), ZSPDIV(:,:), ZSPSCALAR(:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW
  REAL(KIND=JPRB) :: ZCORRECT_VORTICITY_VALUE

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZGP(JPPROMA, 3, IGPBLKS))
  ALLOCATE(ZSPVOR(1,ISPEC2))
  ALLOCATE(ZSPDIV(1,ISPEC2))
  ALLOCATE(ZSPSCALAR(1,ISPEC2))
  ZGP(:,1:2,:) = ROTATIONAL_WIND(IGPBLKS, I_REGIONS_NS, I_REGIONS_EW)
  ZGP(:,3:3,:) = GET_INPUT_FIELD(IMY_PROC, IGPTOTG, IGPBLKS, 1)

  CALL DIR_TRANS(PGP=ZGP, PSPVOR=ZSPVOR, PSPDIV=ZSPDIV, PSPSCALAR=ZSPSCALAR, KPROMA=JPPROMA)

  IF (HAVE_M0_MODE()) THEN
    ! epsilon_{m=0}^{n=1} = sqrt((n^2 - m^2)/(4n^2 - 1)) = sqrt(1/3)
    ZCORRECT_VORTICITY_VALUE = 2.0_JPRB * SQRT(1.0_JPRB / 3.0_JPRB) / PPEARTH_RADIUS

    ! Check all criteria
    KRET = 0

    ! Check all non-(0,0) modes of the scalar field are zero
    IF (.NOT. ALL(APPROX_EQ(ZSPSCALAR(1,2:), 0.0_JPRB))) KRET = 1

    ! Check the (0,0) mode is one
    IF (.NOT. APPROX_EQ(ZSPSCALAR(1,1), 1.0_JPRB)) KRET = 1

    ! Check divergence is all zero
    IF (ANY(.NOT. APPROX_EQ(ZSPDIV, 0.0_JPRB))) KRET = 1

     ! Check all irrelevant vorticity modes are zero
    IF (ANY(.NOT. APPROX_EQ(ZSPVOR(1,1:2), 0.0_JPRB))) KRET = 1

    ! Check selected harmonic has the correct value
    IF (.NOT. APPROX_EQ(ZSPVOR(1,3), ZCORRECT_VORTICITY_VALUE)) KRET = 1
  ELSE
    IF (ALL(APPROX_EQ(ZSPVOR, 0.0_JPRB)) .AND. ALL(APPROX_EQ(ZSPDIV, 0.0_JPRB)) .AND. &
      & ALL(APPROX_EQ(ZSPSCALAR, 0.0_JPRB))) THEN
      KRET = 0
    ELSE
      KRET = 1
    END IF
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPSCALAR, ZSPDIV, ZSPVOR, ZGP)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_1_WIND_1_SCALAR_1

!---------------------------------------------------------------------------------------------------

! Test DIR_TRANS with call mode 2 and just one "3A" scalar field
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_PGP3A_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGP3A(:,:,:,:), ZSPSC3A(:,:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZGP3A(JPPROMA,1,1,IGPBLKS))
  ZGP3A(:,1,:,:) = GET_INPUT_FIELD(IMY_PROC, IGPTOTG, IGPBLKS, 1)
  ALLOCATE(ZSPSC3A(1,ISPEC2,1))

  CALL DIR_TRANS(PGP3A=ZGP3A, PSPSC3A=ZSPSC3A, KPROMA=JPPROMA)

  ! Check only the (0,0) mode (global mean) is one and all the rest are zero
  IF (HAVE_M0_MODE()) THEN
    KRET = MERGE(0, 1, APPROX_EQ(ZSPSC3A(1,1,1), 1.0_JPRB) .AND. &
      &         ALL(APPROX_EQ(ZSPSC3A(1,2:,1), 0.0_JPRB)))
  ELSE
    KRET = MERGE(0, 1, ALL(APPROX_EQ(ZSPSC3A, 0.0_JPRB)))
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPSC3A, ZGP3A)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_PGP3A_1

!---------------------------------------------------------------------------------------------------

! Test DIR_TRANS with call mode 2 and just one "3B" scalar field
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_PGP3B_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGP3B(:,:,:,:), ZSPSC3B(:,:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZGP3B(JPPROMA,1,1,IGPBLKS))
  ZGP3B(:,1,:,:) = GET_INPUT_FIELD(IMY_PROC, IGPTOTG, IGPBLKS, 1)
  ALLOCATE(ZSPSC3B(1,ISPEC2,1))

  CALL DIR_TRANS(PGP3B=ZGP3B, PSPSC3B=ZSPSC3B, KPROMA=JPPROMA)

  ! Check only the (0,0) mode (global mean) is one and all the rest are zero
  IF (HAVE_M0_MODE()) THEN
    KRET = MERGE(0, 1, APPROX_EQ(ZSPSC3B(1,1,1), 1.0_JPRB) .AND. &
      &         ALL(APPROX_EQ(ZSPSC3B(1,2:,1), 0.0_JPRB)))
  ELSE
    KRET = MERGE(0, 1, ALL(APPROX_EQ(ZSPSC3B, 0.0_JPRB)))
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPSC3B, ZGP3B)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_PGP3B_1

!---------------------------------------------------------------------------------------------------

! Test DIR_TRANS with call mode 2 and 1-level wind fields
INTEGER FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_WIND_1() RESULT(KRET) BIND(C)
  REAL(KIND=JPRB), ALLOCATABLE :: ZGPUV(:,:,:,:), ZSPVOR(:,:), ZSPDIV(:,:)
  INTEGER(KIND=JPIM) :: ISPEC2, IGPTOTG, IGPTOT, IMY_PROC, IGPBLKS, I_REGIONS_NS, I_REGIONS_EW
  REAL(KIND=JPRB) :: ZCORRECT_VORTICITY_VALUE

  ! Set up everything
  CALL SETUP_TEST(ISPEC2, IGPTOTG, IGPTOT, IGPBLKS, IMY_PROC, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZGPUV(JPPROMA,1,2,IGPBLKS))
  ZGPUV(:,1,:,:) = ROTATIONAL_WIND(IGPBLKS, I_REGIONS_NS, I_REGIONS_EW)
  ALLOCATE(ZSPVOR(1,ISPEC2))
  ALLOCATE(ZSPDIV(1,ISPEC2))

  CALL DIR_TRANS(PGPUV=ZGPUV, PSPVOR=ZSPVOR, PSPDIV=ZSPDIV, KPROMA=JPPROMA)

  IF (HAVE_M0_MODE()) THEN
    ! epsilon_{m=0}^{n=1} = sqrt((n^2 - m^2)/(4n^2 - 1)) = sqrt(1/3)
    ZCORRECT_VORTICITY_VALUE = 2.0_JPRB * SQRT(1.0_JPRB / 3.0_JPRB) / PPEARTH_RADIUS

    ! Check all criteria
    KRET = 0

    ! Check divergence is all zero
    IF (ANY(.NOT. APPROX_EQ(ZSPDIV, 0.0_JPRB))) KRET = 1

     ! Check all irrelevant vorticity modes are zero
    IF (ANY(.NOT. APPROX_EQ(ZSPVOR(1,1:2), 0.0_JPRB))) KRET = 1

    ! Check selected harmonic has the correct value
    IF (.NOT. APPROX_EQ(ZSPVOR(1,3), ZCORRECT_VORTICITY_VALUE)) KRET = 1
  ELSE
    KRET = MERGE(0, 1, ALL(APPROX_EQ(ZSPVOR, 0.0_JPRB) .AND. ALL(APPROX_EQ(ZSPDIV, 0.0_JPRB))))
  ENDIF

  IF (LUSE_MPI) CALL MPL_ALLREDUCE(KRET, CDOPER="MAX")

  ! Tear down everything
  DEALLOCATE(ZSPDIV, ZSPVOR, ZGPUV)
  CALL CLEANUP_TEST
END FUNCTION ECTRANS_TEST_TRANS_API_DIR_TRANS_CALL_MODE_2_WIND_1

!---------------------------------------------------------------------------------------------------

END MODULE DIR_TRANS_TEST_SUITE
