#define ALIGN(I, A) (((I)+(A)-1)/(A)*(A))
! (C) Copyright 1995- ECMWF.
! (C) Copyright 1995- Meteo-France.
! (C) Copyright 2022- NVIDIA.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

MODULE TRLTOG_MOD
  USE BUFFERED_ALLOCATOR_MOD, ONLY: ALLOCATION_RESERVATION_HANDLE
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: TRLTOG, TRLTOG_HANDLE, PREPARE_TRLTOG, TRLTOG_CPU

  TYPE TRLTOG_HANDLE
    TYPE(ALLOCATION_RESERVATION_HANDLE) :: HCOMBUFR_AND_COMBUFS
  END TYPE
CONTAINS
  FUNCTION PREPARE_TRLTOG(ALLOCATOR,KF_FS,KF_GP) RESULT(HTRLTOG)
    USE PARKIND_ECTRANS,        ONLY: JPIM, JPRBT, JPIB
    USE TPM_DISTR,              ONLY: D
    USE BUFFERED_ALLOCATOR_MOD, ONLY: BUFFERED_ALLOCATOR, RESERVE
    USE ISO_C_BINDING,          ONLY: C_SIZE_T, C_SIZEOF

    IMPLICIT NONE

    TYPE(BUFFERED_ALLOCATOR), INTENT(INOUT) :: ALLOCATOR
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_GP, KF_FS
    TYPE(TRLTOG_HANDLE) :: HTRLTOG

    REAL(KIND=JPRBT) :: DUMMY

    INTEGER(KIND=JPIB) :: NELEM

    NELEM = 0
    NELEM = NELEM + ALIGN(1_JPIB*KF_GP*D%NGPTOT*C_SIZEOF(DUMMY),128) ! ZCOMBUFR
    NELEM = NELEM + ALIGN(1_JPIB*KF_FS*D%NLENGTF*C_SIZEOF(DUMMY),128) !ZCOMBUFS upper obund

    HTRLTOG%HCOMBUFR_AND_COMBUFS = RESERVE(ALLOCATOR, NELEM, "HTRLTOG%HCOMBUFR_AND_COMBUFS")
  END FUNCTION PREPARE_TRLTOG

  SUBROUTINE TRLTOG(ALLOCATOR,HTRLTOG,PREEL_REAL,KF_FS,KF_GP,KF_UV_G,KF_SCALARS_G,KPTRGP,&
     & KVSETUV,KVSETSC,KVSETSC3A,KVSETSC3B,KVSETSC2,&
     & PGP,PGPUV,PGP3A,PGP3B,PGP2)

    !**** *trltog * - transposition of grid point data from latitudinal
    !   to column structure. This takes place between inverse
    !                 FFT and grid point calculations.
    !                 TRLTOG is the inverse of TRGTOL

    ! Version using CUDA-aware MPI

    !     Purpose.
    !     --------


    !**   Interface.
    !     ----------
    !        *call* *trltog(...)

    !        Explicit arguments :
    !        --------------------
    !           PREEL_REAL    -  Latitudinal data ready for direct FFT (input)
    !           PGP    -  Blocked grid point data    (output)
    !           KVSET    - "v-set" for each field      (input)

    !        Implicit arguments :
    !        --------------------

    !     Method.
    !     -------
    !        See documentation

    !     Externals.
    !     ----------

    !     Reference.
    !     ----------
    !        ECMWF Research Department documentation of the IFS

    !     Author.
    !     -------
    !        MPP Group *ECMWF*

    !     Modifications.
    !     --------------
    !        Original  : 95-10-01
    !        D.Dent    : 97-08-04 Reorganisation to allow NPRTRV
    !                             to differ from NPRGPEW
    !        =99-03-29= Mats Hamrud and Deborah Salmond
    !                   JUMP in FFT's changed to 1
    !                   INDEX introduced and ZCOMBUF not used for same PE
    !         01-11-23  Deborah Salmond and John Hague
    !                   LIMP_NOOLAP Option for non-overlapping message passing
    !                               and buffer packing
    !         01-12-18  Peter Towers
    !                   Improved vector performance of LTOG_PACK,LTOG_UNPACK
    !         03-0-02   G. Radnoti: Call barrier always when nproc>1
    !         08-01-01  G.Mozdzynski: cleanup
    !         09-01-02  G.Mozdzynski: use non-blocking recv and send
    !     ------------------------------------------------------------------

    USE PARKIND_ECTRANS,        ONLY: JPIM, JPRB, JPRBT, JPIB
    USE YOMHOOK,                ONLY: LHOOK, DR_HOOK, JPHOOK
    USE MPL_MODULE,             ONLY: MPL_WAIT, MPL_BARRIER, MPL_ABORT
    USE TPM_GEN,                ONLY: LSYNC_TRANS, NERR, LMPOFF
    USE EQ_REGIONS_MOD,         ONLY: MY_REGION_EW, MY_REGION_NS
    USE TPM_DISTR,              ONLY: D,MYSETV, MYSETW, MTAGLG,NPRCIDS,MYPROC,NPROC,NPRTRW,NPRTRV
    USE PE2SET_MOD,             ONLY: PE2SET
    USE MPL_DATA_MODULE,        ONLY: MPL_COMM_OML
    USE OML_MOD,                ONLY: OML_MY_THREAD
    USE ABORT_TRANS_MOD,        ONLY: ABORT_TRANS
#if ECTRANS_HAVE_MPI
    USE MPI_F08,                ONLY: MPI_COMM, MPI_REQUEST, MPI_REAL4, MPI_REAL8
    ! Missing: MPI_ISEND, MPI_IRECV on purpose due to cray-mpi bug (see https://github.com/ecmwf-ifs/ectrans/pull/157)
#endif
    USE TPM_STATS,              ONLY: GSTATS => GSTATS_NVTX
    USE TPM_TRANS,              ONLY: LDIVGP, LSCDERS, LUVDER, LVORGP, NPROMA
    USE BUFFERED_ALLOCATOR_MOD, ONLY: BUFFERED_ALLOCATOR, ASSIGN_PTR, GET_ALLOCATION
    USE ISO_C_BINDING,          ONLY: C_SIZE_T, C_SIZEOF
    USE OPENACC_EXT,            ONLY: EXT_ACC_ARR_DESC, EXT_ACC_PASS, EXT_ACC_CREATE, &
      &                               EXT_ACC_DELETE
#ifdef ACCGPU
    USE OPENACC,                ONLY: ACC_HANDLE_KIND
#endif

    IMPLICIT NONE

    REAL(KIND=JPRBT),  INTENT(INOUT), POINTER  :: PREEL_REAL(:)
    INTEGER(KIND=JPIM),INTENT(IN)  :: KF_FS,KF_GP
    INTEGER(KIND=JPIM),INTENT(IN)  :: KF_UV_G, KF_SCALARS_G
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KPTRGP(:)
    REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP(:,:,:)
    REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGPUV(:,:,:,:)
    REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3A(:,:,:,:)
    REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3B(:,:,:,:)
    REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP2(:,:,:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KVSETUV(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KVSETSC(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KVSETSC3A(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KVSETSC3B(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KVSETSC2(:)

    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR
    TYPE(TRLTOG_HANDLE) :: HTRLTOG

    ! LOCAL VARIABLES

    REAL(KIND=JPRBT), POINTER :: ZCOMBUFS(:),ZCOMBUFR(:)

    LOGICAL :: LLOCAL_CONTRIBUTION
    INTEGER(KIND=JPIB) :: ISENDTOT (NPROC)
    INTEGER(KIND=JPIB) :: IRECVTOT (NPROC)
    INTEGER(KIND=JPIM) :: ISENDTOT_MPI(NPROC)
    INTEGER(KIND=JPIM) :: IRECVTOT_MPI(NPROC)
    INTEGER(KIND=JPIM) :: IREQ     (NPROC*2)
    INTEGER(KIND=JPIM) :: IRECV_TO_PROC(NPROC)
    INTEGER(KIND=JPIM) :: ISEND_TO_PROC(NPROC)

    INTEGER(KIND=JPIM) :: JFLD, J, JI, J1, J2, JGL, JK, JL, IFLDS, JROC, INR, INS
    INTEGER(KIND=JPIM) :: IFIRSTLAT, ILASTLAT, IFLD, IGL, IGLL,&
                 &ISETA, ISETB, ISETV, ISEND, IRECV, ISETW, IPROC, &
                 &IR, ILOCAL_LAT, ISEND_COUNTS, IRECV_COUNTS, IERROR, II, ILEN, IBUFLENS, IBUFLENR, &
                 &JBLK, ILAT_STRIP
    INTEGER(KIND=JPIB) :: IPOS

    ! Contains FIELD, PARS, LEVS
    INTEGER(KIND=JPIM) :: IGP_OFFSETS(KF_GP,3)
    INTEGER(KIND=JPIM), PARAMETER :: IGP_OFFSETS_UV=1, IGP_OFFSETS_GP2=2, IGP_OFFSETS_GP3A=3, IGP_OFFSETS_GP3B=4
    INTEGER(KIND=JPIM) :: IUVPAR,IGP2PAR,IGP3ALEV,IGP3APAR,IGP3BLEV,IGP3BPAR,IPAROFF,IOFF

    INTEGER(KIND=JPIB) :: IIN_TO_SEND_BUFR(D%NLENGTF,2)
    INTEGER(KIND=JPIM) :: IIN_TO_SEND_BUFR_OFFSET(NPROC), IIN_TO_SEND_BUFR_V
    INTEGER(KIND=JPIM) :: IRECV_FIELD_COUNT(NPRTRV),IRECV_FIELD_COUNT_V
    INTEGER(KIND=JPIM) :: IRECV_WSET_SIZE(NPRTRW),IRECV_WSET_SIZE_V
    INTEGER(KIND=JPIM) :: IRECV_WSET_OFFSET(NPRTRW+1), IRECV_WSET_OFFSET_V
    INTEGER(KIND=JPIB), ALLOCATABLE :: ICOMBUFS_OFFSET(:),ICOMBUFR_OFFSET(:), IFLDA(:,:)
    INTEGER(KIND=JPIB) :: ICOMBUFS_OFFSET_V, ICOMBUFR_OFFSET_V

    INTEGER(KIND=JPIM) :: IVSETUV(KF_UV_G)
    INTEGER(KIND=JPIM) :: IVSETSC(KF_SCALARS_G)
    INTEGER(KIND=JPIM) :: IVSET(KF_GP)
    INTEGER(KIND=JPIM) :: J3,IFGP2,IFGP3A,IFGP3B

    REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
    REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_BAR

    TYPE(EXT_ACC_ARR_DESC) :: ACC_POINTERS(5) ! at most 5 copyins...
    INTEGER(KIND=JPIM) :: ACC_POINTERS_CNT = 0

#if ECTRANS_HAVE_MPI
    TYPE(MPI_COMM) :: LOCAL_COMM
    TYPE(MPI_REQUEST) :: IREQUEST(NPROC*2)
#endif

#ifdef PARKINDTRANS_SINGLE
#define TRLTOG_DTYPE MPI_REAL4
#else
#define TRLTOG_DTYPE MPI_REAL8
#endif
#if ECTRANS_HAVE_MPI
    IF(.NOT. LMPOFF) THEN
      LOCAL_COMM%MPI_VAL = MPL_COMM_OML( OML_MY_THREAD() )
    ENDIF
#endif
    !     ------------------------------------------------------------------

    !*       0.    Some initializations
    !              --------------------
    IF (LHOOK) CALL DR_HOOK('TRLTOG',0,ZHOOK_HANDLE)

    ! Note we have either
    ! - KVSETUV and KVSETSC (with PGP, which has u, v, and scalar fields), or
    ! - KVSETUV, KVSETSC2, KVSETSC3A KVSETSC3B (with PGPUV, GP3A, PGP3B and PGP2)
    ! KVSETs are optionals. Their sizes canalso be inferred from KV_UV_G/KV_SCALARS_G (which
    ! should match PSPXXX and PGPXXX arrays)


    ! We first get the decomposition individually
    IVSETUV(:) = -1
    IF (PRESENT(KVSETUV)) IVSETUV(:) = KVSETUV(:)
    IVSETSC(:)=-1
    IF (PRESENT(KVSETSC)) THEN
      IVSETSC(:) = KVSETSC(:)
    ELSE
      IOFF=0
      IF (PRESENT(KVSETSC2)) THEN
        IVSETSC(IOFF+1:IOFF+SIZE(KVSETSC2))=KVSETSC2(:)
        IOFF = IOFF+SIZE(KVSETSC2)
      ENDIF
      IF (PRESENT(KVSETSC3A)) THEN
        DO J3=1,MERGE(UBOUND(PGP3A,3),UBOUND(PGP3A,3)/3,.NOT. LSCDERS)
          IVSETSC(IOFF+1:IOFF+SIZE(KVSETSC3A))=KVSETSC3A(:)
          IOFF=IOFF+SIZE(KVSETSC3A)
        ENDDO
      ENDIF
      IF (PRESENT(KVSETSC3B)) THEN
        ! If SCDERS is on, the size of PGP is 3X larger because it is
        ! holding various derivatives. The problem is that those are
        ! at different non-contiguous positions, hence we treat them
        ! as separate fields
        DO J3=1,MERGE(UBOUND(PGP3B,3),UBOUND(PGP3B,3)/3,.NOT. LSCDERS)
          IVSETSC(IOFF+1:IOFF+SIZE(KVSETSC3B))=KVSETSC3B(:)
          IOFF=IOFF+SIZE(KVSETSC3B)
        ENDDO
      ENDIF
      IF (IOFF > 0 .AND. IOFF /= KF_SCALARS_G ) THEN
        CALL ABORT_TRANS("TRLTOG: Error in IVSETSC computation")
      ENDIF
    ENDIF

    ! Now from UV and Scalars decomposition we get the full decomposition
    IOFF=0
    IF (KF_UV_G > 0) THEN
      IF (LVORGP) THEN
        IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
        IOFF=IOFF+KF_UV_G
      ENDIF
      IF ( LDIVGP) THEN
        IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
        IOFF=IOFF+KF_UV_G
      ENDIF
      IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
      IOFF=IOFF+KF_UV_G
      IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
      IOFF=IOFF+KF_UV_G
    ENDIF
    IF (KF_SCALARS_G > 0) THEN
      IVSET(IOFF+1:IOFF+KF_SCALARS_G) = IVSETSC(:)
      IOFF=IOFF+KF_SCALARS_G
      IF (LSCDERS) THEN
        IVSET(IOFF+1:IOFF+KF_SCALARS_G) = IVSETSC(:)
        IOFF=IOFF+KF_SCALARS_G
      ENDIF
    ENDIF
    IF (KF_UV_G > 0 .AND. LUVDER) THEN
      IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
      IOFF=IOFF+KF_UV_G
      IVSET(IOFF+1:IOFF+KF_UV_G) = IVSETUV(:)
      IOFF=IOFF+KF_UV_G
    ENDIF
    IF (KF_SCALARS_G > 0) THEN
      IF (LSCDERS) THEN
        IVSET(IOFF+1:IOFF+KF_SCALARS_G) = IVSETSC(:)
        IOFF=IOFF+KF_SCALARS_G
      ENDIF
    ENDIF

    IF (.NOT. PRESENT(PGP)) THEN
      ! This is only relevant if we use the split interface (i.e. not PGP)

      IGP2PAR = 0
      IGP3APAR = 0
      IGP3ALEV = 0
      IGP3BPAR = 0
      IGP3BLEV = 0
      IF (PRESENT(PGP2)) THEN
        IGP2PAR = UBOUND(PGP2,2)
        IF(LSCDERS) IGP2PAR = IGP2PAR/3
      ENDIF
      IF (PRESENT(PGP3A)) THEN
        IGP3ALEV = UBOUND(PGP3A,2)
        IGP3APAR = UBOUND(PGP3A,3)
        IF(LSCDERS) IGP3APAR = IGP3APAR/3
      ENDIF
      IF (PRESENT(PGP3B)) THEN
        IGP3BLEV = UBOUND(PGP3B,2)
        IGP3BPAR = UBOUND(PGP3B,3)
        IF(LSCDERS) IGP3BPAR = IGP3BPAR/3
      ENDIF
      IF (IGP2PAR + IGP3ALEV*IGP3APAR + IGP3BPAR*IGP3BLEV /= KF_SCALARS_G) THEN
        WRITE(NERR,*) IGP2PAR, IGP3APAR, IGP3ALEV, IGP3BPAR, IGP3BLEV
        CALL ABORT_TRANS("INCONSISTENCY IN SCALARS")
      ENDIF

      ! This is only relevant if we use the split interface (i.e. not PGP)
      IUVPAR = 1
      IOFF=1
      IF(LVORGP) THEN
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
        IUVPAR=IUVPAR+1
        IOFF=IOFF+KF_UV_G
      ENDIF

      IF(LDIVGP) THEN
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
        IUVPAR=IUVPAR+1
        IOFF=IOFF+KF_UV_G
      ENDIF

      ! U
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
      IUVPAR=IUVPAR+1
      IOFF=IOFF+KF_UV_G

      ! V
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
      IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
      IUVPAR=IUVPAR+1
      IOFF=IOFF+KF_UV_G

      ! Scalars
      ! PGP2
      IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,1) = IGP_OFFSETS_GP2
      IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,2) = (/(J, J=1,IGP2PAR)/)
      IOFF=IOFF+IGP2PAR
      ! PGP3A
      IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,1) = IGP_OFFSETS_GP3A
      IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,2) = (/(1+J/IGP3ALEV, J=0,IGP3APAR*IGP3ALEV-1)/)
      IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,3) = (/(1+MOD(J,IGP3ALEV), J=0,IGP3APAR*IGP3ALEV-1)/)
      IOFF=IOFF+IGP3APAR*IGP3ALEV
      ! PGP3B
      IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,1) = IGP_OFFSETS_GP3B
      IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,2) = (/(1+J/IGP3BLEV, J=0,IGP3BPAR*IGP3BLEV-1)/)
      IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,3) = (/(1+MOD(J,IGP3BLEV), J=0,IGP3BPAR*IGP3BLEV-1)/)
      IOFF=IOFF+IGP3BPAR*IGP3BLEV

      IF(LSCDERS) THEN
        !Scalars NS Derivatives
        ! PGP2
        IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,1) = IGP_OFFSETS_GP2
        IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,2) = (/(J+IGP2PAR, J=1,IGP2PAR)/)
        IOFF=IOFF+IGP2PAR
        ! PGP3A
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,1) = IGP_OFFSETS_GP3A
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,2) = (/(1+IGP3APAR+J/IGP3ALEV, J=0,IGP3APAR*IGP3ALEV-1)/)
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,3) = (/(1+MOD(J,IGP3ALEV), J=0,IGP3APAR*IGP3ALEV-1)/)
        IOFF=IOFF+IGP3APAR*IGP3ALEV
        ! PGP3B
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,1) = IGP_OFFSETS_GP3B
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,2) = (/(1+IGP3BPAR+J/IGP3BLEV, J=0,IGP3BPAR*IGP3BLEV-1)/)
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,3) = (/(1+MOD(J,IGP3BLEV), J=0,IGP3BPAR*IGP3BLEV-1)/)
        IOFF=IOFF+IGP3BPAR*IGP3BLEV
      ENDIF

      IF(LUVDER) THEN
        ! U Derivative NS
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
        IUVPAR=IUVPAR+1
        IOFF=IOFF+KF_UV_G

        ! V Derivative NS
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,1) = IGP_OFFSETS_UV
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,2) = IUVPAR
        IGP_OFFSETS(IOFF:IOFF+KF_UV_G-1,3) = (/(J, J=1,KF_UV_G)/)
        IUVPAR=IUVPAR+1
        IOFF=IOFF+KF_UV_G
      ENDIF

      IF(LSCDERS) THEN
        !Scalars NS Derivatives
        ! PGP2
        IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,1) = IGP_OFFSETS_GP2
        IGP_OFFSETS(IOFF:IOFF+IGP2PAR-1,2) = (/(J+2*IGP2PAR, J=1,IGP2PAR)/)
        IOFF=IOFF+IGP2PAR
        ! PGP3A
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,1) = IGP_OFFSETS_GP3A
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,2) = (/(1+2*IGP3APAR+J/IGP3ALEV, J=0,IGP3APAR*IGP3ALEV-1)/)
        IGP_OFFSETS(IOFF:IOFF+IGP3APAR*IGP3ALEV-1,3) = (/(1+MOD(J,IGP3ALEV), J=0,IGP3APAR*IGP3ALEV-1)/)
        IOFF=IOFF+IGP3APAR*IGP3ALEV
        ! PGP3B
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,1) = IGP_OFFSETS_GP3B
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,2) = (/(1+2*IGP3BPAR+J/IGP3BLEV, J=0,IGP3BPAR*IGP3BLEV-1)/)
        IGP_OFFSETS(IOFF:IOFF+IGP3BPAR*IGP3BLEV-1,3) = (/(1+MOD(J,IGP3BLEV), J=0,IGP3BPAR*IGP3BLEV-1)/)
        IOFF=IOFF+IGP3BPAR*IGP3BLEV
      ENDIF
    ENDIF

    CALL GSTATS(1806,0)

    ! Prepare receiver arrays
    ! find number of fields on a certain V-set
    IF(NPRTRV == 1) THEN
      ! This is needed because KVSET(JFLD) == -1 if there is only one V-set
      IRECV_FIELD_COUNT(1) = KF_GP
    ELSE
      IRECV_FIELD_COUNT(:) = 0
      DO JFLD=1,KF_GP
        IRECV_FIELD_COUNT(IVSET(JFLD)) = IRECV_FIELD_COUNT(IVSET(JFLD)) + 1
      ENDDO
    ENDIF
    ! find number of grid-points on a certain W-set that overlap with myself
    IRECV_WSET_SIZE(:) = 0
    DO ILOCAL_LAT=D%NFRSTLAT(MY_REGION_NS),D%NLSTLAT(MY_REGION_NS)
      ILAT_STRIP = ILOCAL_LAT-D%NFRSTLAT(MY_REGION_NS)+D%NPTRFLOFF+1
      IRECV_WSET_SIZE(D%NPROCL(ILOCAL_LAT)) = &
          & IRECV_WSET_SIZE(D%NPROCL(ILOCAL_LAT))+D%NONL(ILAT_STRIP,MY_REGION_EW)
    ENDDO
    ! sum up offsets
    IRECV_WSET_OFFSET(1) = 0
    DO JROC=1,NPRTRW
      IRECV_WSET_OFFSET(JROC+1)=IRECV_WSET_OFFSET(JROC)+IRECV_WSET_SIZE(JROC)
    ENDDO
    DO JROC=1,NPROC
      CALL PE2SET(JROC,ISETA,ISETB,ISETW,ISETV)
      ! total recv size is # points per field * # fields
      IRECVTOT(JROC) = 1_JPIB*IRECV_WSET_SIZE(ISETW)*IRECV_FIELD_COUNT(ISETV)
    ENDDO

    ! Prepare sender arrays
    IIN_TO_SEND_BUFR_OFFSET(1) = 0
    DO JROC=1,NPROC
      ! Get new offset to my current KINDEX entry
      IF (JROC > 1 .AND. KF_FS > 0) THEN
        IIN_TO_SEND_BUFR_OFFSET(JROC) = IIN_TO_SEND_BUFR_OFFSET(JROC-1)+ISENDTOT(JROC-1)/KF_FS
      ELSEIF (JROC > 1) THEN
        IIN_TO_SEND_BUFR_OFFSET(JROC) = IIN_TO_SEND_BUFR_OFFSET(JROC-1)
      ENDIF

      CALL PE2SET(JROC,ISETA,ISETB,ISETW,ISETV)

      ! MAX(Index of first fourier latitude for this W set, first latitude of a senders A set)
      ! i.e. we find the overlap between what we have on sender side (others A set) and the receiver
      ! (me, the W-set). Ideally those conincide, at least mostly.
      IFIRSTLAT = MAX(D%NPTRLS(MYSETW),D%NFRSTLAT(ISETA))
      ! MIN(Index of last fourier latitude for this W set, last latitude of a senders A set)
      ILASTLAT  = MIN(D%NPTRLS(MYSETW)+D%NULTPP(MYSETW)-1,D%NLSTLAT(ISETA))

      IPOS = 0
      DO JGL=IFIRSTLAT,ILASTLAT
        ! get from "actual" latitude to the latitude strip offset
        IGL  = JGL-D%NFRSTLAT(ISETA)+D%NPTRFRSTLAT(ISETA)
        ! get from "actual" latitude to the latitude offset
        IGLL = JGL-D%NPTRLS(MYSETW)+1
        DO JL=1,D%NONL(IGL,ISETB)
          IPOS = IPOS+1
          ! offset to first layer of this gridpoint
          IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_OFFSET(JROC)+IPOS,1) = &
              & 1_JPIB*KF_FS*D%NSTAGTF(IGLL)+(D%NSTA(IGL,ISETB)-1)+(JL-1)
          ! distance between two layers of this gridpoint
          IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_OFFSET(JROC)+IPOS,2) = &
              & D%NSTAGTF(IGLL+1)-D%NSTAGTF(IGLL)
        ENDDO
      ENDDO
      !we always receive the full fourier space
      ISENDTOT(JROC) = IPOS*KF_FS
    ENDDO
    LLOCAL_CONTRIBUTION = ISENDTOT(MYPROC) > 0

#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC DATA COPYIN(IGP_OFFSETS) ASYNC(1)
#endif

    ACC_POINTERS_CNT = 0
    IF (PRESENT(PGP)) THEN
      ACC_POINTERS_CNT = ACC_POINTERS_CNT + 1
      ACC_POINTERS(ACC_POINTERS_CNT) = EXT_ACC_PASS(PGP)
    ENDIF
    IF (PRESENT(PGPUV)) THEN
      ACC_POINTERS_CNT = ACC_POINTERS_CNT + 1
      ACC_POINTERS(ACC_POINTERS_CNT) = EXT_ACC_PASS(PGPUV)
    ENDIF
    IF (PRESENT(PGP2)) THEN
      ACC_POINTERS_CNT = ACC_POINTERS_CNT + 1
      ACC_POINTERS(ACC_POINTERS_CNT) = EXT_ACC_PASS(PGP2)
    ENDIF
    IF (PRESENT(PGP3A)) THEN
      ACC_POINTERS_CNT = ACC_POINTERS_CNT + 1
      ACC_POINTERS(ACC_POINTERS_CNT) = EXT_ACC_PASS(PGP3A)
    ENDIF
    IF (PRESENT(PGP3B)) THEN
      ACC_POINTERS_CNT = ACC_POINTERS_CNT + 1
      ACC_POINTERS(ACC_POINTERS_CNT) = EXT_ACC_PASS(PGP3B)
    ENDIF

    IF (ACC_POINTERS_CNT > 0) CALL EXT_ACC_CREATE(ACC_POINTERS(1:ACC_POINTERS_CNT), &
#ifdef ACCGPU
         & STREAM=1_ACC_HANDLE_KIND)
#endif
#ifdef OMPGPU
         & STREAM=1)
#endif

#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC DATA IF(PRESENT(PGP))   PRESENT(PGP) ASYNC(1)
    !$ACC DATA IF(PRESENT(PGPUV)) PRESENT(PGPUV) ASYNC(1)
    !$ACC DATA IF(PRESENT(PGP2))  PRESENT(PGP2) ASYNC(1)
    !$ACC DATA IF(PRESENT(PGP3A)) PRESENT(PGP3A) ASYNC(1)
    !$ACC DATA IF(PRESENT(PGP3B)) PRESENT(PGP3B) ASYNC(1)

    ! Present until self contribution and packing are done
    !$ACC DATA COPYIN(IIN_TO_SEND_BUFR) PRESENT(PREEL_REAL) IF(KF_FS > 0) ASYNC(1)
#endif
#ifdef OMPGPU
#endif

    CALL GSTATS(1806,1)
    
    ! Figure out processes that send or recv something
    ISEND_COUNTS   = 0
    IRECV_COUNTS   = 0
    DO JROC=1,NPROC
      IF( JROC /= MYPROC) THEN
        IF(IRECVTOT(JROC) > 0) THEN
          ! I have to recv something, so let me store that
          IRECV_COUNTS = IRECV_COUNTS + 1
          IRECV_TO_PROC(IRECV_COUNTS)=JROC
        ENDIF
        IF(ISENDTOT(JROC) > 0) THEN
          ! I have to send something, so let me store that
          ISEND_COUNTS = ISEND_COUNTS+1
          ISEND_TO_PROC(ISEND_COUNTS)=JROC
        ENDIF
      ENDIF
    ENDDO

    ! ... build this data structure now during the MPI communication
    ! Allocate this buffer now. Add 1 for self contribution
    ALLOCATE(IFLDA(KF_GP,1+IRECV_COUNTS))

    ! Copy local contribution
    IF(LLOCAL_CONTRIBUTION) THEN
      ! I have to send something to myself...

      ! Input is KF_GP fields. We find the resulting KF_FS fields.
      IFLDS = 0
      DO JFLD=1,KF_GP
        IF(IVSET(JFLD) == MYSETV .OR. IVSET(JFLD) == -1) THEN
          IFLDS = IFLDS+1
          IF(PRESENT(KPTRGP)) THEN
            IFLDA(IFLDS,1) = KPTRGP(JFLD)
          ELSE
            IFLDA(IFLDS,1) = JFLD
          ENDIF
        ENDIF
      ENDDO
    ENDIF

    DO INR=1,IRECV_COUNTS
      IRECV=IRECV_TO_PROC(INR)
      CALL PE2SET(IRECV,ISETA,ISETB,ISETW,ISETV)
      IFLDS = 0
      DO JFLD=1,KF_GP
        IF(IVSET(JFLD) == ISETV .OR. IVSET(JFLD) == -1 ) THEN
          IFLDS = IFLDS+1
          IF(PRESENT(KPTRGP)) THEN
            IFLDA(IFLDS,1+INR)=KPTRGP(JFLD)
          ELSE
            IFLDA(IFLDS,1+INR)=JFLD
          ENDIF
        ENDIF
      ENDDO
    ENDDO
   
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC DATA COPYIN(IFLDA) ASYNC(1)
#endif

    ! Copy local contribution
    IF(LLOCAL_CONTRIBUTION) THEN

      CALL GSTATS(1604,0)

      IRECV_WSET_OFFSET_V = IRECV_WSET_OFFSET(MYSETW)
      IRECV_WSET_SIZE_V = IRECV_WSET_SIZE(MYSETW)
      IIN_TO_SEND_BUFR_V = IIN_TO_SEND_BUFR_OFFSET(MYPROC)
      IF (PRESENT(PGP)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
        !$ACC PARALLEL LOOP COLLAPSE(2) DEFAULT(NONE) PRIVATE(JK,JBLK,IFLD,IPOS) &
        !$ACC&         FIRSTPRIVATE(KF_FS,IRECV_WSET_SIZE_V,IRECV_WSET_OFFSET_V, &
        !$ACC&         IIN_TO_SEND_BUFR_V,NPROMA) ASYNC(1)
#endif
        DO JFLD=1,KF_FS
          DO JL=1,IRECV_WSET_SIZE_V
            JK = MOD(IRECV_WSET_OFFSET_V+JL-1,NPROMA)+1
            JBLK = (IRECV_WSET_OFFSET_V+JL-1)/NPROMA+1
            IFLD = IFLDA(JFLD,1)
            IPOS = IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,1)+ &
                & (JFLD-1)*IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,2)+1
            PGP(JK,IFLD,JBLK) = PREEL_REAL(IPOS)
          ENDDO
        ENDDO
      ELSE
#ifdef OMPGPU
#endif
#ifdef ACCGPU
        !$ACC PARALLEL LOOP COLLAPSE(2) DEFAULT(NONE) PRIVATE(JK,JBLK,IFLD,IPOS) &
        !$ACC&              FIRSTPRIVATE(KF_FS,IRECV_WSET_SIZE_V,IRECV_WSET_OFFSET_V, &
        !$ACC&              IIN_TO_SEND_BUFR_V,NPROMA) ASYNC(1)
#endif
        DO JFLD=1,KF_FS
          DO JL=1,IRECV_WSET_SIZE_V
            JK = MOD(IRECV_WSET_OFFSET_V+JL-1,NPROMA)+1
            JBLK = (IRECV_WSET_OFFSET_V+JL-1)/NPROMA+1
            IFLD = IFLDA(JFLD,1)
            IPOS = IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,1)+ &
                & (JFLD-1)*IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,2)+1
            IF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_UV) THEN
              PGPUV(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = PREEL_REAL(IPOS)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP2) THEN
              PGP2(JK,IGP_OFFSETS(IFLD,2),JBLK)=PREEL_REAL(IPOS)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP3A) THEN
              PGP3A(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = PREEL_REAL(IPOS)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP3B) THEN
              PGP3B(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = PREEL_REAL(IPOS)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
      CALL GSTATS(1604,1)
    ENDIF

    ALLOCATE(ICOMBUFS_OFFSET(ISEND_COUNTS+1))
    ICOMBUFS_OFFSET(1) = 0
    DO JROC=1,ISEND_COUNTS
      ICOMBUFS_OFFSET(JROC+1) = ICOMBUFS_OFFSET(JROC) + ISENDTOT(ISEND_TO_PROC(JROC))
    ENDDO
    ALLOCATE(ICOMBUFR_OFFSET(IRECV_COUNTS+1))
    ICOMBUFR_OFFSET(1) = 0
    DO JROC=1,IRECV_COUNTS
      ICOMBUFR_OFFSET(JROC+1) = ICOMBUFR_OFFSET(JROC) + IRECVTOT(IRECV_TO_PROC(JROC))
    ENDDO

    IF (IRECV_COUNTS > 0) THEN
      CALL ASSIGN_PTR(ZCOMBUFR, GET_ALLOCATION(ALLOCATOR, HTRLTOG%HCOMBUFR_AND_COMBUFS),&
          & 1_JPIB, ICOMBUFR_OFFSET(IRECV_COUNTS+1)*C_SIZEOF(ZCOMBUFR(1)))
    ENDIF
    IF (ISEND_COUNTS > 0) THEN
      CALL ASSIGN_PTR(ZCOMBUFS, GET_ALLOCATION(ALLOCATOR, HTRLTOG%HCOMBUFR_AND_COMBUFS),&
          & ALIGN(1_JPIB*KF_GP*D%NGPTOT*C_SIZEOF(ZCOMBUFR(1)),128)+1, &
          & ICOMBUFS_OFFSET(ISEND_COUNTS+1)*C_SIZEOF(ZCOMBUFS(1)))
    ENDIF

#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC DATA PRESENT(ZCOMBUFS) IF(ISEND_COUNTS > 0) ASYNC(1)
#endif
    CALL GSTATS(1605,0)
    DO INS=1,ISEND_COUNTS
      IPROC = ISEND_TO_PROC(INS)
      ILEN = ISENDTOT(IPROC)/KF_FS
      IIN_TO_SEND_BUFR_V = IIN_TO_SEND_BUFR_OFFSET(IPROC)
      ICOMBUFS_OFFSET_V = ICOMBUFS_OFFSET(INS)
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC PARALLEL LOOP DEFAULT(NONE) PRIVATE(IPOS) FIRSTPRIVATE(KF_FS,ILEN,IIN_TO_SEND_BUFR_V, &
      !$ACC&              ICOMBUFS_OFFSET_V) COLLAPSE(2) ASYNC(1)
#endif
      DO JFLD=1,KF_FS
        DO JL=1,ILEN
          IPOS = IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,1)+ &
              & (JFLD-1)*IIN_TO_SEND_BUFR(IIN_TO_SEND_BUFR_V+JL,2)+1
          ZCOMBUFS(ICOMBUFS_OFFSET_V+(JFLD-1)*ILEN+JL) = PREEL_REAL(IPOS)
        ENDDO
      ENDDO
    ENDDO
    CALL GSTATS(1605,1)
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC END DATA ! ZCOMBUFS

    !$ACC WAIT(1)
#endif

    CALL GSTATS(805,0)

    IF (LSYNC_TRANS) THEN
      CALL GSTATS(440,0)
      CALL MPL_BARRIER(CDSTRING='')
      CALL GSTATS(440,1)
    ENDIF
    CALL GSTATS(421,0)

    IR=0
    !...Receive loop.........................................................
#ifdef USE_GPU_AWARE_MPI
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC HOST_DATA USE_DEVICE(ZCOMBUFS,ZCOMBUFR)
#endif
#else
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !! this is safe-but-slow fallback for running without GPU-aware MPI
    !$ACC UPDATE HOST(ZCOMBUFS) IF(ISEND_COUNTS > 0)
#endif
#endif

    ! Skip the own contribution because this is ok to overflow
    ISENDTOT(MYPROC) = 0
    IRECVTOT(MYPROC) = 0

    ISENDTOT_MPI = ISENDTOT
    IRECVTOT_MPI = IRECVTOT
    IF (ANY(ISENDTOT_MPI /= ISENDTOT)) &
      & CALL MPL_ABORT("Overflow in trltog")
    IF (ANY(IRECVTOT_MPI /= IRECVTOT)) &
      & CALL MPL_ABORT("Overflow in trltog")

    DO INR=1,IRECV_COUNTS
      IR=IR+1
      IRECV=IRECV_TO_PROC(INR)
#if ECTRANS_HAVE_MPI
      CALL MPI_IRECV(ZCOMBUFR(ICOMBUFR_OFFSET(INR)+1:ICOMBUFR_OFFSET(INR+1)), &
        & IRECVTOT_MPI(IRECV), &
        & TRLTOG_DTYPE,NPRCIDS(IRECV)-1, &
        & MTAGLG, LOCAL_COMM, IREQUEST(IR), &
        & IERROR )
      IREQ(IR) = IREQUEST(IR)%MPI_VAL
#else
      CALL ABORT_TRANS("Should not be here: MPI is disabled")
#endif
    ENDDO

    !...Send loop.........................................................
    DO INS=1,ISEND_COUNTS
      IR=IR+1
      ISEND=ISEND_TO_PROC(INS)
#if ECTRANS_HAVE_MPI
      CALL MPI_ISEND(ZCOMBUFS(ICOMBUFS_OFFSET(INS)+1:ICOMBUFS_OFFSET(INS+1)),ISENDTOT_MPI(ISEND), &
        & TRLTOG_DTYPE, NPRCIDS(ISEND)-1,MTAGLG,LOCAL_COMM,IREQUEST(IR),IERROR)
      IREQ(IR) = IREQUEST(IR)%MPI_VAL
#else
        CALL ABORT_TRANS("Should not be here: MPI is disabled")
#endif
    ENDDO

    IF(IR > 0) THEN
      CALL MPL_WAIT(KREQUEST=IREQ(1:IR), &
      & CDSTRING='TRLTOG: WAIT FOR SENDS AND RECEIVES')
    ENDIF

#ifdef USE_GPU_AWARE_MPI
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC END HOST_DATA
#endif
#else
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !! this is safe-but-slow fallback for running without GPU-aware MPI
    !$ACC UPDATE DEVICE(ZCOMBUFR) IF(IRECV_COUNTS > 0)
#endif
#endif

    IF (LSYNC_TRANS) THEN
      CALL GSTATS(441,0)
      CALL MPL_BARRIER(CDSTRING='')
      CALL GSTATS(441,1)
    ENDIF
    CALL GSTATS(421,1)

#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC DATA PRESENT(ZCOMBUFR) IF(IRECV_COUNTS > 0) ASYNC(1)
#endif
    CALL GSTATS(805,1)

    !  Unpack loop.........................................................

    CALL GSTATS(1606,0)
    DO INR=1,IRECV_COUNTS
      IRECV=IRECV_TO_PROC(INR)
      CALL PE2SET(IRECV,ISETA,ISETB,ISETW,ISETV)

      IRECV_FIELD_COUNT_V = IRECV_FIELD_COUNT(ISETV)
      ICOMBUFR_OFFSET_V = ICOMBUFR_OFFSET(INR)

      IRECV_WSET_OFFSET_V = IRECV_WSET_OFFSET(ISETW)
      IRECV_WSET_SIZE_V = IRECV_WSET_SIZE(ISETW)
      IF (PRESENT(PGP)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
        !$ACC PARALLEL LOOP COLLAPSE(2) DEFAULT(NONE) PRIVATE(JK,JBLK,IFLD,JI) &
        !$ACC&              FIRSTPRIVATE(INR,IRECV_FIELD_COUNT_V,IRECV_WSET_SIZE_V,&
        !$ACC&              IRECV_WSET_OFFSET_V,NPROMA,ICOMBUFR_OFFSET_V) ASYNC(1)
#endif
        DO JFLD=1,IRECV_FIELD_COUNT_V
          DO JL=1,IRECV_WSET_SIZE_V
            JK = MOD(IRECV_WSET_OFFSET_V+JL-1,NPROMA)+1
            JBLK = (IRECV_WSET_OFFSET_V+JL-1)/NPROMA+1
            IFLD=IFLDA(JFLD,1+INR)
            JI = ICOMBUFR_OFFSET_V+(JFLD-1)*IRECV_WSET_SIZE_V+JL
            PGP(JK,IFLD,JBLK) = ZCOMBUFR(JI)
          ENDDO
        ENDDO
      ELSE
#ifdef OMPGPU
#endif
#ifdef ACCGPU
        !$ACC PARALLEL LOOP COLLAPSE(2) DEFAULT(NONE) PRIVATE(JK,JBLK,IFLD,JI) &
        !$ACC&              FIRSTPRIVATE(INR,IRECV_FIELD_COUNT_V,IRECV_WSET_SIZE_V, &
        !$ACC&              IRECV_WSET_OFFSET_V,NPROMA,ICOMBUFR_OFFSET_V) ASYNC(1)
#endif
        DO JFLD=1,IRECV_FIELD_COUNT_V
          DO JL=1,IRECV_WSET_SIZE_V
            JK = MOD(IRECV_WSET_OFFSET_V+JL-1,NPROMA)+1
            JBLK = (IRECV_WSET_OFFSET_V+JL-1)/NPROMA+1
            IFLD=IFLDA(JFLD,1+INR)
            JI = ICOMBUFR_OFFSET_V+(JFLD-1)*IRECV_WSET_SIZE_V+JL
            IF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_UV) THEN
              PGPUV(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = ZCOMBUFR(JI)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP2) THEN
              PGP2(JK,IGP_OFFSETS(IFLD,2),JBLK) = ZCOMBUFR(JI)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP3A) THEN
              PGP3A(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = ZCOMBUFR(JI)
            ELSEIF(IGP_OFFSETS(IFLD,1) == IGP_OFFSETS_GP3B) THEN
              PGP3B(JK,IGP_OFFSETS(IFLD,3),IGP_OFFSETS(IFLD,2),JBLK) = ZCOMBUFR(JI)
            ENDIF
          ENDDO
        ENDDO
      ENDIF
    ENDDO
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC WAIT(1)
#endif

#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC END DATA ! ZOMBUFR
#endif
    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      !$ACC WAIT(1)
#endif
      CALL GSTATS(440,0)
      CALL MPL_BARRIER(CDSTRING='')
      CALL GSTATS(440,1)
    ENDIF
    CALL GSTATS(422,0)
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC END DATA ! IFLDA
    !$ACC END DATA ! PREEL_REAL
    !$ACC END DATA ! PGP3B
    !$ACC END DATA ! PGP3A
    !$ACC END DATA ! PGP2
    !$ACC END DATA ! PGPUV
    !$ACC END DATA ! PGP
#endif
    IF (PRESENT(PGP)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC UPDATE HOST(PGP) ASYNC(1)
#endif
    ENDIF
    IF (PRESENT(PGPUV)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC UPDATE HOST(PGPUV) ASYNC(1)
#endif
    ENDIF
    IF (PRESENT(PGP2)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC UPDATE HOST(PGP2) ASYNC(1)
#endif
    ENDIF
    IF (PRESENT(PGP3A)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC UPDATE HOST(PGP3A) ASYNC(1)
#endif
    ENDIF
    IF (PRESENT(PGP3B)) THEN
#ifdef OMPGPU
#endif
#ifdef ACCGPU
      !$ACC UPDATE HOST(PGP3B) ASYNC(1)
#endif
    ENDIF
    IF (ACC_POINTERS_CNT > 0) CALL EXT_ACC_DELETE(ACC_POINTERS(1:ACC_POINTERS_CNT), &
#ifdef ACCGPU
         & STREAM=1_ACC_HANDLE_KIND)
#endif
#ifdef OMPGPU
         & STREAM=1)
#endif

    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      !$ACC WAIT(1)
#endif
      CALL GSTATS(442,0)
      CALL MPL_BARRIER(CDSTRING='')
      CALL GSTATS(442,1)
    ENDIF
    CALL GSTATS(422,1)
#ifdef OMPGPU
#endif
#ifdef ACCGPU
    !$ACC END DATA ! IRECVBUFR_TO_OUT,PGPINDICES

    !$ACC WAIT(1)
#endif

    CALL GSTATS(1606,1)

    ! Free this now
    DEALLOCATE(IFLDA)

    IF (LHOOK) CALL DR_HOOK('TRLTOG',1,ZHOOK_HANDLE)
  END SUBROUTINE TRLTOG
SUBROUTINE TRLTOG_CPU(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,KPTRGP,&
 &PGP,PGPUV,PGP3A,PGP3B,PGP2)

!**** *TRLTOG * - head routine for transposition of grid point data from latitudinal
!                 to column structure (this takes place between inverse
!                 FFT and grid point calculations)
!                 TRLTOG is the inverse of TRGTOL

!**   Interface.
!     ----------
!        *call* *TRLTOG(...)

!        Explicit arguments :
!        --------------------
!           PGLAT    -  Latitudinal data ready for direct FFT (input)
!           PGP    -  Blocked grid point data    (output)
!           KVSET    - "v-set" for each field      (input)

!        Implicit arguments :
!        --------------------

!     Method.
!     -------
!        See documentation

!     Externals.
!     ----------

!     Reference.
!     ----------
!        ECMWF Research Department documentation of the IFS

!     Author.
!     -------
!        R. El Khatib *Meteo-France*

!     Modifications.
!     --------------
!        Original  : 18-Aug-2014 from trltog
!        R. El Khatib 09-Sep-2020 NSTACK_MEMORY_TR
!     ------------------------------------------------------------------

USE PARKIND1  ,ONLY : JPIM     ,JPRB
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK, JPHOOK

USE TPM_GEN         ,ONLY : NSTACK_MEMORY_TR
USE TPM_DISTR       ,ONLY : D, NPRTRNS, NPROC
USE TPM_TRANS       ,ONLY : NGPBLKS

IMPLICIT NONE

INTEGER(KIND=JPIM),INTENT(IN)  :: KF_FS,KF_GP
INTEGER(KIND=JPIM),INTENT(IN)  :: KF_SCALARS_G
REAL(KIND=JPRB),INTENT(IN)     :: PGLAT(KF_FS,D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KVSET(KF_GP)
INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KPTRGP(:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP(:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGPUV(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3A(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3B(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP2(:,:,:)

INTEGER(KIND=JPIM) :: ISENDCOUNT
INTEGER(KIND=JPIM) :: IRECVCOUNT
INTEGER(KIND=JPIM) :: INSEND
INTEGER(KIND=JPIM) :: INRECV
INTEGER(KIND=JPIM) :: ISENDTOT (NPROC)
INTEGER(KIND=JPIM) :: IRECVTOT (NPROC)
INTEGER(KIND=JPIM) :: ISEND    (NPROC)
INTEGER(KIND=JPIM) :: IRECV    (NPROC)
INTEGER(KIND=JPIM) :: IINDEX(D%NLENGTF)
INTEGER(KIND=JPIM) :: INDOFF(NPROC)
INTEGER(KIND=JPIM) :: IGPTRSEND(2,NGPBLKS,NPRTRNS)
INTEGER(KIND=JPIM) :: ISETAL(NPROC), ISETBL(NPROC), ISETWL(NPROC), ISETVL(NPROC)

REAL(KIND=JPHOOK) :: ZHOOK_HANDLE

!     ------------------------------------------------------------------

IF (LHOOK) CALL DR_HOOK('TRLTOG',0,ZHOOK_HANDLE)

CALL TRLTOG_PROLOG(KF_FS,KF_GP,KVSET,&
 & ISENDCOUNT,IRECVCOUNT,INSEND,INRECV,ISENDTOT,IRECVTOT,ISEND,IRECV,IINDEX,INDOFF,IGPTRSEND, &
 & ISETAL,ISETBL,ISETWL,ISETVL)
IF (NSTACK_MEMORY_TR==0) THEN
  CALL TRLTOG_COMM_HEAP(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET, &
   & ISENDCOUNT,IRECVCOUNT,INSEND,INRECV,ISENDTOT,IRECVTOT,ISEND,IRECV,IINDEX,INDOFF,IGPTRSEND, &
   & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2,&
   & ISETAL,ISETBL,ISETWL,ISETVL)
ELSE
  CALL TRLTOG_COMM_STACK(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET, &
   & ISENDCOUNT,IRECVCOUNT,INSEND,INRECV,ISENDTOT,IRECVTOT,ISEND,IRECV,IINDEX,INDOFF,IGPTRSEND, &
   & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2,&
   & ISETAL,ISETBL,ISETWL,ISETVL)
ENDIF

IF (LHOOK) CALL DR_HOOK('TRLTOG',1,ZHOOK_HANDLE)

!     ------------------------------------------------------------------

END SUBROUTINE TRLTOG_CPU

SUBROUTINE TRLTOG_PROLOG(KF_FS,KF_GP,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND, &
 & KSETAL,KSETBL,KSETWL,KSETVL)

!**** *TRLTOG_PROLOG * - prolog for transposition of grid point data from latitudinal
!                 to column structure (this takes place between inverse
!                 FFT and grid point calculations) : the purpose is essentially 
!                 to compute the size of communication buffers in order to enable
!                 the use of automatic arrays later.
!                 TRLTOG_PROLOG is the inverse of TRGTOL_PROLOG

!     Purpose.
!     --------

!**   Interface.
!     ----------
!        *call* *TRLTOG_PROLOG(...)

!        Explicit arguments :
!        --------------------
!           KVSET    - "v-set" for each field      (input)

!        Implicit arguments :
!        --------------------

!     Method.
!     -------
!        See documentation

!     Externals.
!     ----------

!     Reference.
!     ----------
!        ECMWF Research Department documentation of the IFS

!     Author.
!     -------
!        R. El Khatib *Meteo-France*

!     Modifications.
!     --------------
!        Original  : 18-Aug-2014 from trltog
!     ------------------------------------------------------------------

USE PARKIND1  ,ONLY : JPIM

USE TPM_DISTR       ,ONLY : D, MYSETW, NPRTRNS, MYPROC, NPROC
USE TPM_TRANS       ,ONLY : NGPBLKS

USE INIGPTR_MOD     ,ONLY : INIGPTR
USE PE2SET_MOD      ,ONLY : PE2SET
!

IMPLICIT NONE

INTEGER(KIND=JPIM),INTENT(IN)  :: KF_FS,KF_GP
INTEGER(KIND=JPIM),INTENT(IN)  :: KVSET(KF_GP)

INTEGER(KIND=JPIM), INTENT(OUT) :: KSENDCOUNT
INTEGER(KIND=JPIM), INTENT(OUT) :: KRECVCOUNT
INTEGER(KIND=JPIM), INTENT(OUT) :: KNSEND
INTEGER(KIND=JPIM), INTENT(OUT) :: KNRECV
INTEGER(KIND=JPIM), INTENT(OUT) :: KSENDTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KRECVTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KSEND    (NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KRECV    (NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KINDEX(D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(OUT) :: KNDOFF(NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KGPTRSEND(2,NGPBLKS,NPRTRNS)
INTEGER(KIND=JPIM), INTENT(OUT) :: KSETAL(NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KSETBL(NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KSETWL(NPROC)
INTEGER(KIND=JPIM), INTENT(OUT) :: KSETVL(NPROC)

INTEGER(KIND=JPIM) :: IGPTRRECV(NPRTRNS)
INTEGER(KIND=JPIM) :: IFIRSTLAT, IGL, IGLL, ILASTLAT, IPOS, ISETA, ISETB, ISETV
INTEGER(KIND=JPIM) :: ISEND, JFLD, JGL, JL, ISETW, JROC, J
INTEGER(KIND=JPIM) :: INDOFFX,IBUFLENS,IBUFLENR

!     ------------------------------------------------------------------

!*       0.    Some initializations
!              --------------------

CALL GSTATS(1806,0)

CALL INIGPTR(KGPTRSEND,IGPTRRECV)

INDOFFX  = 0
IBUFLENS = 0
IBUFLENR = 0
KNRECV = 0
KNSEND = 0

DO JROC=1,NPROC

  CALL PE2SET(JROC,KSETAL(JROC),KSETBL(JROC),KSETWL(JROC),KSETVL(JROC))
  ISEND      = JROC
  ISETA=KSETAL(JROC)
  ISETB=KSETBL(JROC)
  ISETW=KSETWL(JROC)
  ISETV=KSETVL(JROC)
!             count up expected number of fields
  IPOS = 0
  DO JFLD=1,KF_GP
    IF(KVSET(JFLD) == ISETV .OR. KVSET(JFLD) == -1) IPOS = IPOS+1
  ENDDO
  KRECVTOT(JROC) = IGPTRRECV(ISETW)*IPOS
  IF(KRECVTOT(JROC) > 0 .AND. MYPROC /= JROC) THEN
    KNRECV = KNRECV + 1
    KRECV(KNRECV)=JROC
  ENDIF

  IF( JROC /= MYPROC) IBUFLENR = MAX(IBUFLENR,KRECVTOT(JROC))

  IFIRSTLAT = MAX(D%NPTRLS(MYSETW),D%NFRSTLAT(ISETA))
  ILASTLAT  = MIN(D%NPTRLS(MYSETW)+D%NULTPP(MYSETW)-1,D%NLSTLAT(ISETA))

  IPOS = 0
  DO JGL=IFIRSTLAT,ILASTLAT
    IGL  = D%NPTRFRSTLAT(ISETA)+JGL-D%NFRSTLAT(ISETA)
    IPOS = IPOS+D%NONL(IGL,ISETB)
  ENDDO

  KSENDTOT(JROC) = IPOS*KF_FS
  IF( JROC /= MYPROC) THEN
    IBUFLENS = MAX(IBUFLENS,KSENDTOT(JROC))
    IF(KSENDTOT(JROC) > 0) THEN
      KNSEND = KNSEND+1
      KSEND(KNSEND)=JROC
    ENDIF
  ENDIF

  IF(IPOS > 0) THEN
    KNDOFF(JROC) = INDOFFX
    INDOFFX = INDOFFX+IPOS
    IPOS = 0
    DO JGL=IFIRSTLAT,ILASTLAT
      IGL  = D%NPTRFRSTLAT(ISETA)+JGL-D%NFRSTLAT(ISETA)
      IGLL = JGL-D%NPTRLS(MYSETW)+1
      DO JL=D%NSTA(IGL,ISETB)+D%NSTAGTF(IGLL),&
       &D%NSTA(IGL,ISETB)+D%NSTAGTF(IGLL)+D%NONL(IGL,ISETB)-1
        IPOS = IPOS+1
        KINDEX(IPOS+KNDOFF(JROC)) = JL
      ENDDO
    ENDDO
  ENDIF
ENDDO

KSENDCOUNT=0
KRECVCOUNT=0
DO J=1,NPROC
  KSENDCOUNT=MAX(KSENDCOUNT,KSENDTOT(J))
  KRECVCOUNT=MAX(KRECVCOUNT,KRECVTOT(J))
ENDDO

CALL GSTATS(1806,1)

END SUBROUTINE TRLTOG_PROLOG

SUBROUTINE TRLTOG_COMM_HEAP(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND,&
 & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2, &
 & KSETAL, KSETBL,KSETWL,KSETVL)

USE PARKIND1  ,ONLY : JPIM     ,JPRB
USE TPM_DISTR ,ONLY : D, NPRTRNS, NPROC
USE TPM_TRANS ,ONLY : NGPBLKS

IMPLICIT NONE

INTEGER(KIND=JPIM), INTENT(IN) :: KF_FS,KF_GP
REAL(KIND=JPRB),INTENT(IN)     :: PGLAT(KF_FS,D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KVSET(KF_GP)
INTEGER(KIND=JPIM), INTENT(IN) :: KF_SCALARS_G
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KNSEND
INTEGER(KIND=JPIM), INTENT(IN) :: KNRECV
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSEND    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECV    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KINDEX(D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KNDOFF(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KGPTRSEND(2,NGPBLKS,NPRTRNS)
INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KPTRGP(:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP(:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGPUV(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3A(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3B(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP2(:,:,:)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETAL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETBL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETWL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETVL(NPROC)

REAL(KIND=JPRB), ALLOCATABLE, SAVE :: ZCOMBUFS_HEAP(:,:)
REAL(KIND=JPRB), ALLOCATABLE, SAVE :: ZCOMBUFR_HEAP(:,:)
INTEGER(KIND=JPIM), SAVE :: INRECV_PREV = -1
INTEGER(KIND=JPIM), SAVE :: INSEND_PREV = -1
INTEGER(KIND=JPIM), SAVE :: IRECVCOUNT_PREV = -1
INTEGER(KIND=JPIM), SAVE :: ISENDCOUNT_PREV = -1

IF ( .NOT. ALLOCATED(ZCOMBUFS_HEAP) ) THEN
  ALLOCATE(ZCOMBUFS_HEAP(-1:KSENDCOUNT,KNSEND))
  ISENDCOUNT_PREV = KSENDCOUNT
  INSEND_PREV = KNSEND
ELSEIF ( KSENDCOUNT .NE. ISENDCOUNT_PREV .OR. KNSEND .NE. INSEND_PREV ) THEN
  DEALLOCATE(ZCOMBUFS_HEAP)
  ALLOCATE(ZCOMBUFS_HEAP(-1:KSENDCOUNT,KNSEND))
  ISENDCOUNT_PREV = KSENDCOUNT
  INSEND_PREV = KNSEND
ENDIF

! Now, force the OS to allocate this shared array right now, not when it starts to be used which is
! an OPEN-MP loop, that would cause a threads synchronization lock :
IF (KNSEND > 0 .AND. KSENDCOUNT >= -1) ZCOMBUFS_HEAP(-1,1) = HUGE(1._JPRB)

IF ( .NOT. ALLOCATED(ZCOMBUFR_HEAP) ) THEN
  ALLOCATE(ZCOMBUFR_HEAP(-1:KRECVCOUNT,KNRECV))
  IRECVCOUNT_PREV = KRECVCOUNT
  INRECV_PREV = KNRECV
ELSEIF ( KRECVCOUNT .NE. IRECVCOUNT_PREV .OR. KNRECV .NE. INRECV_PREV ) THEN
  DEALLOCATE(ZCOMBUFR_HEAP)
  ALLOCATE(ZCOMBUFR_HEAP(-1:KRECVCOUNT,KNRECV))
  IRECVCOUNT_PREV = KRECVCOUNT
  INRECV_PREV = KNRECV
ENDIF

CALL TRLTOG_COMM(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND,&
 & ZCOMBUFS_HEAP,ZCOMBUFR_HEAP, &
 & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2, &
 & KSETAL, KSETBL,KSETWL,KSETVL)

END SUBROUTINE TRLTOG_COMM_HEAP

SUBROUTINE TRLTOG_COMM_STACK(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND,&
 & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2, &
 & KSETAL, KSETBL,KSETWL,KSETVL)

USE PARKIND1  ,ONLY : JPIM     ,JPRB
USE TPM_DISTR ,ONLY : D, NPRTRNS, NPROC
USE TPM_TRANS ,ONLY : NGPBLKS

IMPLICIT NONE

INTEGER(KIND=JPIM), INTENT(IN) :: KF_FS,KF_GP
REAL(KIND=JPRB),INTENT(IN)     :: PGLAT(KF_FS,D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KVSET(KF_GP)
INTEGER(KIND=JPIM), INTENT(IN) :: KF_SCALARS_G
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KNSEND
INTEGER(KIND=JPIM), INTENT(IN) :: KNRECV
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSEND    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECV    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KINDEX(D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KNDOFF(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KGPTRSEND(2,NGPBLKS,NPRTRNS)
INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KPTRGP(:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP(:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGPUV(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3A(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3B(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP2(:,:,:)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETAL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETBL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETWL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETVL(NPROC)

REAL(KIND=JPRB) :: ZCOMBUFS_STACK(-1:KSENDCOUNT,KNSEND)
REAL(KIND=JPRB) :: ZCOMBUFR_STACK(-1:KRECVCOUNT,KNRECV)

CALL TRLTOG_COMM(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND,&
 & ZCOMBUFS_STACK,ZCOMBUFR_STACK, &
 & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2, &
 & KSETAL, KSETBL,KSETWL,KSETVL)

END SUBROUTINE TRLTOG_COMM_STACK

SUBROUTINE TRLTOG_COMM(PGLAT,KF_FS,KF_GP,KF_SCALARS_G,KVSET,&
 & KSENDCOUNT,KRECVCOUNT,KNSEND,KNRECV,KSENDTOT,KRECVTOT,KSEND,KRECV,KINDEX,KNDOFF,KGPTRSEND,&
 & PCOMBUFS,PCOMBUFR, &
 & KPTRGP,PGP,PGPUV,PGP3A,PGP3B,PGP2, &
 & KSETAL, KSETBL,KSETWL,KSETVL)
 

!**** *trltog * - transposition of grid point data from latitudinal
!                 to column structure. This takes place between inverse
!                 FFT and grid point calculations.
!                 TRLTOG_COMM is the inverse of TRGTOL

!     Purpose.
!     --------

!**   Interface.
!     ----------
!        *call* *trltog(...)

!        Explicit arguments :
!        --------------------
!           PGLAT    -  Latitudinal data ready for direct FFT (input)
!           PGP    -  Blocked grid point data    (output)
!           KVSET    - "v-set" for each field      (input)

!        Implicit arguments :
!        --------------------

!     Method.
!     -------
!        See documentation

!     Externals.
!     ----------

!     Reference.
!     ----------
!        ECMWF Research Department documentation of the IFS

!     Author.
!     -------
!        MPP Group *ECMWF*

!     Modifications.
!     --------------
!        Original  : 95-10-01
!        D.Dent    : 97-08-04 Reorganisation to allow NPRTRV
!                             to differ from NPRGPEW
!        =99-03-29= Mats Hamrud and Deborah Salmond
!                   JUMP in FFT's changed to 1
!                   KINDEX introduced and PCOMBUF not used for same PE
!         01-11-23  Deborah Salmond and John Hague
!                   LIMP_NOOLAP Option for non-overlapping message passing
!                               and buffer packing
!         01-12-18  Peter Towers
!                   Improved vector performance of LTOG_PACK,LTOG_UNPACK
!         03-0-02   G. Radnoti: Call barrier always when nproc>1
!         08-01-01  G.Mozdzynski: cleanup
!         09-01-02  G.Mozdzynski: use non-blocking recv and send
!        R. El Khatib 09-Sep-2020 64 bits addressing for PGLAT
!     ------------------------------------------------------------------

USE PARKIND1  ,ONLY : JPIM     ,JPRB    ,JPIB
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK, JPHOOK

USE MPL_MODULE  ,ONLY : MPL_RECV, MPL_SEND, MPL_WAIT, JP_NON_BLOCKING_STANDARD, MPL_WAITANY, &
     & JP_BLOCKING_STANDARD, MPL_BARRIER, JP_BLOCKING_BUFFERED

USE TPM_GEN         ,ONLY : NOUT, NTRANS_SYNC_LEVEL
USE TPM_DISTR       ,ONLY : D, MYSETV, MYSETW, MTAGLG,      &
     &                      NPRCIDS, NPRTRNS, MYPROC, NPROC
USE TPM_TRANS       ,ONLY : LDIVGP, LSCDERS, LUVDER, LVORGP, NGPBLKS

USE PE2SET_MOD      ,ONLY : PE2SET
USE ABORT_TRANS_MOD ,ONLY : ABORT_TRANS

IMPLICIT NONE


INTEGER(KIND=JPIM), INTENT(IN) :: KF_FS,KF_GP
REAL(KIND=JPRB),INTENT(IN)     :: PGLAT(KF_FS,D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KVSET(KF_GP)
INTEGER(KIND=JPIM), INTENT(IN) :: KF_SCALARS_G
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVCOUNT
INTEGER(KIND=JPIM), INTENT(IN) :: KNSEND
INTEGER(KIND=JPIM), INTENT(IN) :: KNRECV
INTEGER(KIND=JPIM), INTENT(IN) :: KSENDTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECVTOT (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSEND    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KRECV    (NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KINDEX(D%NLENGTF)
INTEGER(KIND=JPIM), INTENT(IN) :: KNDOFF(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KGPTRSEND(2,NGPBLKS,NPRTRNS)
REAL(KIND=JPRB), INTENT(INOUT) :: PCOMBUFS(-1:KSENDCOUNT,KNSEND)
REAL(KIND=JPRB), INTENT(INOUT) :: PCOMBUFR(-1:KRECVCOUNT,KNRECV)
INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN) :: KPTRGP(:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP(:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGPUV(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3A(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP3B(:,:,:,:)
REAL(KIND=JPRB),OPTIONAL,INTENT(OUT)     :: PGP2(:,:,:)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETAL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETBL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETWL(NPROC)
INTEGER(KIND=JPIM), INTENT(IN) :: KSETVL(NPROC)

! LOCAL VARIABLES

INTEGER(KIND=JPIM) :: IPOSPLUS(KNRECV)
INTEGER(KIND=JPIM) :: ISETW(KNRECV)
INTEGER(KIND=JPIM) :: JPOS(NGPBLKS,KNRECV)
INTEGER(KIND=JPIM) :: IFLDA(KF_GP,KNRECV)
INTEGER(KIND=JPIM) :: IREQ_SEND(NPROC)
INTEGER(KIND=JPIM) :: IREQ_RECV(NPROC)

INTEGER(KIND=JPIM) :: IFIRST, IFLD, ILAST, IPOS, ISETA, ISETB, IRECV, ISETV
INTEGER(KIND=JPIM) :: ISEND, ITAG,  JBLK, JFLD, JK, JL, IFLDS, INR, INS
INTEGER(KIND=JPIM) :: II,ILEN, IFLDT, JI, JJ, J

INTEGER(KIND=JPIB) :: JFLD64

LOGICAL   :: LLPGPUV,LLPGP3A,LLPGP3B,LLPGP2,LLPGPONLY
LOGICAL   :: LLUV(KF_GP),LLGP2(KF_GP),LLGP3A(KF_GP),LLGP3B(KF_GP)
LOGICAL   :: LLINDER
INTEGER(KIND=JPIM) :: IUVLEVS(KF_GP),IUVPARS(KF_GP),IGP2PARS(KF_GP)
INTEGER(KIND=JPIM) :: IGP3APARS(KF_GP),IGP3ALEVS(KF_GP),IGP3BPARS(KF_GP),IGP3BLEVS(KF_GP)
INTEGER(KIND=JPIM) :: IUVPAR,IUVLEV,IGP2PAR,IGP3ALEV,IGP3APAR,IGP3BLEV,IGP3BPAR,IPAROFF
INTEGER(KIND=JPIM) :: IOFF,IOFF1,IOFFNS,IOFFEW,J1,J2, JNR
INTEGER(KIND=JPIM) :: IFLDOFF(KF_FS)
INTEGER(KIND=JPIM) :: IRECV_FLD_START,IRECV_FLD_END
INTEGER(KIND=JPIM) :: ISEND_FLD_START,ISEND_FLD_END
INTEGER(KIND=JPIM) :: IGPTROFF(NGPBLKS)

REAL(KIND=JPHOOK) :: ZHOOK_HANDLE_BAR

!     ------------------------------------------------------------------

!*       0.    Some initializations
!              --------------------

ITAG   = MTAGLG

IF (LHOOK) CALL DR_HOOK('TRLTOG_BAR',0,ZHOOK_HANDLE_BAR)
CALL GSTATS_BARRIER(762)
IF (LHOOK) CALL DR_HOOK('TRLTOG_BAR',1,ZHOOK_HANDLE_BAR)

CALL GSTATS(805,0)

IF (NTRANS_SYNC_LEVEL <= 0) THEN
   !...Receive loop.........................................................
   DO INR=1,KNRECV
      IRECV=KRECV(INR)
      CALL MPL_RECV(PCOMBUFR(-1:KRECVTOT(IRECV),INR), &
           & KSOURCE=NPRCIDS(IRECV), &
           & KMP_TYPE=JP_NON_BLOCKING_STANDARD,KREQUEST=IREQ_RECV(INR), &
           & KTAG=ITAG,CDSTRING='TRLTOG_COMM: NON-BLOCKING IRECV' )
   ENDDO
ENDIF

CALL GSTATS(805,1)

CALL GSTATS(1806,0)
LLINDER = .FALSE.
LLPGPUV = .FALSE.
LLPGP3A = .FALSE.
LLPGP3B = .FALSE.
LLPGP2  = .FALSE.
LLPGPONLY = .FALSE.
IF(PRESENT(KPTRGP))  LLINDER = .TRUE.
IF(PRESENT(PGP))     LLPGPONLY=.TRUE.
IF(PRESENT(PGPUV))   LLPGPUV=.TRUE.
IF(PRESENT(PGP3A))   LLPGP3A=.TRUE.
IF(PRESENT(PGP3B))   LLPGP3B=.TRUE.
IF(PRESENT(PGP2))    LLPGP2=.TRUE.

IUVPAR=0
IUVLEV=0
IOFF1=0
IOFFNS=KF_SCALARS_G
IOFFEW=2*KF_SCALARS_G

LLUV(:) = .FALSE.
IF (LLPGPUV) THEN
  IOFF=0
  IUVLEV=UBOUND(PGPUV,2)
  IF(LVORGP) THEN
    IUVPAR=IUVPAR+1
    DO J=1,IUVLEV
      IUVLEVS(IOFF+J)=J
      IUVPARS(IOFF+J)=IUVPAR
      LLUV(IOFF+J)=.TRUE.
    ENDDO
    IOFF=IOFF+IUVLEV
  ENDIF
  IF(LDIVGP) THEN
    IUVPAR=IUVPAR+1
    DO J=1,IUVLEV
      IUVLEVS(IOFF+J)=J
      IUVPARS(IOFF+J)=IUVPAR
      LLUV(IOFF+J)=.TRUE.
    ENDDO
    IOFF=IOFF+IUVLEV
  ENDIF
  DO J=1,IUVLEV
    IUVLEVS(IOFF+J)=J
    IUVPARS(IOFF+J)=IUVPAR+1
    IUVLEVS(IOFF+J+IUVLEV)=J
    IUVPARS(IOFF+J+IUVLEV)=IUVPAR+2
  ENDDO
  IUVPAR=IUVPAR+2
  LLUV(IOFF+1:IOFF+2*IUVLEV)=.TRUE.
  IOFF=IOFF+2*IUVLEV
  IOFF1=IOFF
  IOFFNS=IOFFNS+IOFF
  IOFFEW=IOFFEW+IOFF

  IOFF=IUVPAR*IUVLEV+KF_SCALARS_G
  IF(LUVDER) THEN
    IF(LSCDERS) IOFF=IOFF+KF_SCALARS_G
    DO J=1,IUVLEV
      IUVLEVS(IOFF+J)=J
      IUVPARS(IOFF+J)=IUVPAR+1
      LLUV(IOFF+J)=.TRUE.
      IUVLEVS(IOFF+J+IUVLEV)=J
      IUVPARS(IOFF+J+IUVLEV)=IUVPAR+2
      LLUV(IOFF+J+IUVLEV)=.TRUE.
    ENDDO
    IUVPAR=IUVPAR+2
    IOFF=IOFF+2*IUVLEV
    IOFFEW=IOFFEW+2*IUVLEV
  ENDIF
ENDIF

LLGP2(:)=.FALSE.
IF(LLPGP2) THEN
  IOFF=IOFF1
  IGP2PAR=UBOUND(PGP2,2)
  IF(LSCDERS) IGP2PAR=IGP2PAR/3
  DO J=1,IGP2PAR
    LLGP2(J+IOFF) = .TRUE.
    IGP2PARS(J+IOFF)=J
  ENDDO
  IOFF1=IOFF1+IGP2PAR
  IF(LSCDERS) THEN
    IOFF=IOFFNS
    DO J=1,IGP2PAR
      LLGP2(J+IOFF) = .TRUE.
      IGP2PARS(J+IOFF)=J+IGP2PAR
    ENDDO
    IOFFNS=IOFF+IGP2PAR
    IOFF=IOFFEW
    DO J=1,IGP2PAR
      LLGP2(J+IOFF) = .TRUE.
      IGP2PARS(J+IOFF)=J+2*IGP2PAR
    ENDDO
    IOFFEW=IOFF+IGP2PAR
  ENDIF
ENDIF

LLGP3A(:) = .FALSE.
IF(LLPGP3A) THEN
  IGP3ALEV=UBOUND(PGP3A,2)
  IGP3APAR=UBOUND(PGP3A,3)
  IF(LSCDERS) IGP3APAR=IGP3APAR/3
  IOFF=IOFF1
  DO J1=1,IGP3APAR
    DO J2=1,IGP3ALEV
      LLGP3A(J2+(J1-1)*IGP3ALEV+IOFF) = .TRUE.
      IGP3APARS(J2+(J1-1)*IGP3ALEV+IOFF)=J1
      IGP3ALEVS(J2+(J1-1)*IGP3ALEV+IOFF)=J2
    ENDDO
  ENDDO
  IPAROFF=IGP3APAR
  IOFF1=IOFF1+IGP3APAR*IGP3ALEV
  IF(LSCDERS) THEN
    IOFF=IOFFNS
    DO J1=1,IGP3APAR
      DO J2=1,IGP3ALEV
        LLGP3A(J2+(J1-1)*IGP3ALEV+IOFF) = .TRUE.
        IGP3APARS(J2+(J1-1)*IGP3ALEV+IOFF)=J1+IPAROFF
        IGP3ALEVS(J2+(J1-1)*IGP3ALEV+IOFF)=J2
      ENDDO
    ENDDO
    IPAROFF=IPAROFF+IGP3APAR
    IOFFNS=IOFFNS+IGP3APAR*IGP3ALEV
    IOFF=IOFFEW
    DO J1=1,IGP3APAR
      DO J2=1,IGP3ALEV
        LLGP3A(J2+(J1-1)*IGP3ALEV+IOFF) = .TRUE.
        IGP3APARS(J2+(J1-1)*IGP3ALEV+IOFF)=J1+IPAROFF
        IGP3ALEVS(J2+(J1-1)*IGP3ALEV+IOFF)=J2
      ENDDO
    ENDDO
    IOFFEW=IOFFEW+IGP3APAR*IGP3ALEV
  ENDIF
ENDIF

LLGP3B(:) = .FALSE.
IF(LLPGP3B) THEN
  IGP3BLEV=UBOUND(PGP3B,2)
  IGP3BPAR=UBOUND(PGP3B,3)
  IF(LSCDERS) IGP3BPAR=IGP3BPAR/3
  IOFF=IOFF1
  DO J1=1,IGP3BPAR
    DO J2=1,IGP3BLEV
      LLGP3B(J2+(J1-1)*IGP3BLEV+IOFF) = .TRUE.
      IGP3BPARS(J2+(J1-1)*IGP3BLEV+IOFF)=J1
      IGP3BLEVS(J2+(J1-1)*IGP3BLEV+IOFF)=J2
    ENDDO
  ENDDO
  IPAROFF=IGP3BPAR
  IOFF1=IOFF1+IGP3BPAR*IGP3BLEV
  IF(LSCDERS) THEN
    IOFF=IOFFNS
    DO J1=1,IGP3BPAR
      DO J2=1,IGP3BLEV
        LLGP3B(J2+(J1-1)*IGP3BLEV+IOFF) = .TRUE.
        IGP3BPARS(J2+(J1-1)*IGP3BLEV+IOFF)=J1+IPAROFF
        IGP3BLEVS(J2+(J1-1)*IGP3BLEV+IOFF)=J2
      ENDDO
    ENDDO
    IPAROFF=IPAROFF+IGP3BPAR
    IOFFNS=IOFFNS+IGP3BPAR*IGP3BLEV
    IOFF=IOFFEW
    DO J1=1,IGP3BPAR
      DO J2=1,IGP3BLEV
        LLGP3B(J2+(J1-1)*IGP3BLEV+IOFF) = .TRUE.
        IGP3BPARS(J2+(J1-1)*IGP3BLEV+IOFF)=J1+IPAROFF
        IGP3BLEVS(J2+(J1-1)*IGP3BLEV+IOFF)=J2
      ENDDO
    ENDDO
    IOFFEW=IOFFEW+IGP3BPAR*IGP3BLEV
  ENDIF
ENDIF  

CALL GSTATS(1806,1)


! Copy local contribution
IF( KRECVTOT(MYPROC) > 0 )THEN
  IFLDS = 0
  DO JFLD=1,KF_GP
    IF(KVSET(JFLD) == MYSETV .OR. KVSET(JFLD) == -1) THEN
      IFLDS = IFLDS+1
      IF(LLINDER) THEN
        IFLDOFF(IFLDS) = KPTRGP(JFLD)
      ELSE
        IFLDOFF(IFLDS) = JFLD
      ENDIF
    ENDIF
  ENDDO

  IPOS=0
  DO JBLK=1,NGPBLKS
    IGPTROFF(JBLK)=IPOS
    IFIRST = KGPTRSEND(1,JBLK,MYSETW)
    IF(IFIRST > 0) THEN
      ILAST = KGPTRSEND(2,JBLK,MYSETW)
      IPOS=IPOS+ILAST-IFIRST+1
    ENDIF
  ENDDO

  CALL GSTATS(1604,0)
#ifdef __NEC__
! Loops inversion is still better on Aurora machines, according to CHMI. REK.
!$OMP PARALLEL DO SCHEDULE(DYNAMIC) PRIVATE(JFLD64,JBLK,JK,IFLD,IPOS,IFIRST,ILAST)
#else
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(JFLD64,JBLK,JK,IFLD,IPOS,IFIRST,ILAST)
#endif
  DO JBLK=1,NGPBLKS
    IFIRST = KGPTRSEND(1,JBLK,MYSETW)
    IF(IFIRST > 0) THEN
      ILAST = KGPTRSEND(2,JBLK,MYSETW)
! Address PGLAT over 64 bits because its size may exceed 2 GB for big data and
! small number of tasks.
      IF(LLPGPONLY) THEN
        DO JFLD64=1,IFLDS
          IFLD = IFLDOFF(JFLD64)
!DIR$ VECTOR ALWAYS
          DO JK=IFIRST,ILAST
            IPOS = KNDOFF(MYPROC)+IGPTROFF(JBLK)+JK-IFIRST+1
            PGP(JK,IFLD,JBLK) = PGLAT(JFLD64,KINDEX(IPOS))
          ENDDO
        ENDDO
      ELSE
        DO JFLD64=1,IFLDS
          IFLD = IFLDOFF(JFLD64)
          IF(LLUV(IFLD)) THEN
!DIR$ VECTOR ALWAYS
            DO JK=IFIRST,ILAST
              IPOS = KNDOFF(MYPROC)+IGPTROFF(JBLK)+JK-IFIRST+1
              PGPUV(JK,IUVLEVS(IFLD),IUVPARS(IFLD),JBLK) = PGLAT(JFLD64,KINDEX(IPOS))
            ENDDO
          ELSEIF(LLGP2(IFLD)) THEN
!DIR$ VECTOR ALWAYS
            DO JK=IFIRST,ILAST
              IPOS = KNDOFF(MYPROC)+IGPTROFF(JBLK)+JK-IFIRST+1
              PGP2(JK,IGP2PARS(IFLD),JBLK)=PGLAT(JFLD64,KINDEX(IPOS))
            ENDDO
          ELSEIF(LLGP3A(IFLD)) THEN
!DIR$ VECTOR ALWAYS
            DO JK=IFIRST,ILAST
              IPOS = KNDOFF(MYPROC)+IGPTROFF(JBLK)+JK-IFIRST+1
              PGP3A(JK,IGP3ALEVS(IFLD),IGP3APARS(IFLD),JBLK)=PGLAT(JFLD64,KINDEX(IPOS))
            ENDDO
          ELSEIF(LLGP3B(IFLD)) THEN
!DIR$ VECTOR ALWAYS
            DO JK=IFIRST,ILAST
              IPOS = KNDOFF(MYPROC)+IGPTROFF(JBLK)+JK-IFIRST+1
              PGP3B(JK,IGP3BLEVS(IFLD),IGP3BPARS(IFLD),JBLK)=PGLAT(JFLD64,KINDEX(IPOS))
            ENDDO
          ELSE
            WRITE(NOUT,*)'TRLTOG_MOD: ERROR',JFLD64,IFLD
            CALL ABORT_TRANS('TRLTOG_MOD: ERROR')
          ENDIF
        ENDDO
      ENDIF
    ENDIF
  ENDDO
!$OMP END PARALLEL DO
  CALL GSTATS(1604,1)

ENDIF

!
! loop over the number of processors we need to communicate with.
! NOT MYPROC
!
! Now overlapping buffer packing/unpacking with sends/waits
! Time as if all communications to avoid double accounting

CALL GSTATS(805,0)

!  Pack+send loop.........................................................

ISEND_FLD_START = 1
ISEND_FLD_END   = KF_FS
DO INS=1,KNSEND
  ISEND=KSEND(INS)
  ILEN = KSENDTOT(ISEND)/KF_FS
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(JFLD,JL,II)
  DO JL=1,ILEN
    II = KINDEX(KNDOFF(ISEND)+JL)
    DO JFLD=ISEND_FLD_START,ISEND_FLD_END
      PCOMBUFS((JFLD-ISEND_FLD_START)*ILEN+JL,INS) = PGLAT(JFLD,II)
    ENDDO
  ENDDO
!$OMP END PARALLEL DO
  PCOMBUFS(-1,INS) = 1
  PCOMBUFS(0,INS)  = KF_FS
  IF (NTRANS_SYNC_LEVEL <= 1) THEN
     CALL MPL_SEND(PCOMBUFS(-1:KSENDTOT(ISEND),INS),KDEST=NPRCIDS(ISEND),&
          & KMP_TYPE=JP_NON_BLOCKING_STANDARD,KREQUEST=IREQ_SEND(INS), &
          & KTAG=ITAG,CDSTRING='TRLTOG_COMM: NON-BLOCKING ISEND')
  ELSE
     CALL MPL_SEND(PCOMBUFS(-1:KSENDTOT(ISEND),INS),KDEST=NPRCIDS(ISEND),&
          & KMP_TYPE=JP_BLOCKING_BUFFERED, &
          & KTAG=ITAG,CDSTRING='TRLTOG_COMM: BLOCKING BUFFERED BSEND')
  ENDIF
ENDDO

!  Unpack loop.........................................................

!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(INR,IRECV,ISETA,ISETB,ISETV,IFLD,JFLD,IPOS,JBLK,IFIRST,ILAST)
DO INR=1,KNRECV
  IRECV=KRECV(INR)
  
  ISETA=KSETAL(IRECV)
  ISETB=KSETBL(IRECV)
  ISETW(INR)=KSETWL(IRECV)
  ISETV=KSETVL(IRECV)
  IFLD = 0
  DO JFLD=1,KF_GP
    IF(KVSET(JFLD) == ISETV .OR. KVSET(JFLD) == -1 ) THEN
      IFLD = IFLD+1
      IFLDA(IFLD,INR)=JFLD
    ENDIF
  ENDDO
  IPOS = 0
  IPOSPLUS(INR)=0
  DO JBLK=1,NGPBLKS
    IFIRST = KGPTRSEND(1,JBLK,ISETW(INR))
    IF(IFIRST > 0) THEN
      ILAST = KGPTRSEND(2,JBLK,ISETW(INR))
      JPOS(JBLK,INR)=IPOS
      IPOSPLUS(INR)=IPOSPLUS(INR)+(ILAST-IFIRST+1)
      IPOS=IPOS+(ILAST-IFIRST+1)
    ENDIF
  ENDDO
ENDDO
!$OMP END PARALLEL DO

DO JNR=1,KNRECV
  
  IF (NTRANS_SYNC_LEVEL <= 0) THEN
     CALL MPL_WAITANY(KREQUEST=IREQ_RECV(1:KNRECV),KINDEX=INR,&
          & CDSTRING='TRLTOG_COMM: WAIT FOR ANY RECEIVES')
  ELSE
     INR = JNR
     IRECV=KRECV(INR)
     CALL MPL_RECV(PCOMBUFR(-1:KRECVTOT(IRECV),INR), &
          & KSOURCE=NPRCIDS(IRECV), &
          & KMP_TYPE=JP_BLOCKING_STANDARD, &
          & KTAG=ITAG,CDSTRING='TRLTOG_COMM: BLOCKING RECV' )
  ENDIF

  IPOS=IPOSPLUS(INR)
  IRECV_FLD_START = PCOMBUFR(-1,INR)
  IRECV_FLD_END   = PCOMBUFR(0,INR)
  
!$OMP PARALLEL DO SCHEDULE(STATIC) PRIVATE(IFLDT,IFIRST,ILAST,JK,JJ,JI,JBLK)
  DO JJ=IRECV_FLD_START,IRECV_FLD_END
    IFLDT=IFLDA(JJ,INR)
    DO JBLK=1,NGPBLKS
      IFIRST = KGPTRSEND(1,JBLK,ISETW(INR))
      IF(IFIRST > 0) THEN
        ILAST = KGPTRSEND(2,JBLK,ISETW(INR))
        IF(LLINDER) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGP(JK,KPTRGP(IFLDT),JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ELSEIF(LLPGPONLY) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGP(JK,IFLDT,JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ELSEIF(LLUV(IFLDT)) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGPUV(JK,IUVLEVS(IFLDT),IUVPARS(IFLDT),JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ELSEIF(LLGP2(IFLDT)) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGP2(JK,IGP2PARS(IFLDT),JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ELSEIF(LLGP3A(IFLDT)) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGP3A(JK,IGP3ALEVS(IFLDT),IGP3APARS(IFLDT),JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ELSEIF(LLGP3B(IFLDT)) THEN
          DO JK=IFIRST,ILAST
            JI=(JJ-IRECV_FLD_START)*IPOS+JPOS(JBLK,INR)+JK-IFIRST+1
            PGP3B(JK,IGP3BLEVS(IFLDT),IGP3BPARS(IFLDT),JBLK) = PCOMBUFR(JI,INR)
          ENDDO
        ENDIF
      ENDIF
    ENDDO
  ENDDO
!$OMP END PARALLEL DO
ENDDO

IF (NTRANS_SYNC_LEVEL <= 1) THEN
   IF(KNSEND > 0) THEN
      CALL MPL_WAIT(KREQUEST=IREQ_SEND(1:KNSEND),CDSTRING='TRLTOG_COMM: WAIT FOR ISENDS')
   ENDIF
ENDIF

IF (NTRANS_SYNC_LEVEL >= 1) THEN
   CALL MPL_BARRIER(CDSTRING='TRLTOG_COMM: BARRIER AT END')
ENDIF

CALL GSTATS(805,1)

CALL GSTATS_BARRIER2(762)

END SUBROUTINE TRLTOG_COMM
END MODULE TRLTOG_MOD

