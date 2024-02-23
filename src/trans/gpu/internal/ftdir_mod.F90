! (C) Copyright 2000- ECMWF.
! (C) Copyright 2000- Meteo-France.
! (C) Copyright 2022- NVIDIA.
! 
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

MODULE FTDIR_MOD
  USE BUFFERED_ALLOCATOR_MOD
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: FTDIR, FTDIR_HANDLE, PREPARE_FTDIR

  TYPE FTDIR_HANDLE
  END TYPE
CONTAINS

  FUNCTION PREPARE_FTDIR() RESULT(HFTDIR)
    IMPLICIT NONE
    TYPE(FTDIR_HANDLE) :: HFTDIR
  END FUNCTION

  SUBROUTINE FTDIR(ALLOCATOR,HFTDIR,PREEL_REAL,PREEL_COMPLEX,KFIELD)
    !**** *FTDIR - Direct Fourier transform

    !     Purpose. Routine for Grid-point to Fourier transform
    !     --------

    !**   Interface.
    !     ----------
    !        CALL FTDIR(..)

    !        Explicit arguments :  PREEL   - Fourier/grid-point array
    !        --------------------  KFIELD   - number of fields

    !     Method.
    !     -------

    !     Externals.  FFT992 - FFT routine
    !     ----------
    !

    !     Author.
    !     -------
    !        Mats Hamrud *ECMWF*

    !     Modifications.
    !     --------------
    !        Original : 00-03-03
    !        G. Radnoti 01-04-24 2D model (NLOEN=1)
    !        D. Degrauwe  (Feb 2012): Alternative extension zone (E')
    !        G. Mozdzynski (Oct 2014): support for FFTW transforms
    !        G. Mozdzynski (Jun 2015): Support alternative FFTs to FFTW
    !     ------------------------------------------------------------------

    USE TPM_GEN         ,ONLY : LSYNC_TRANS
    USE PARKIND_ECTRANS ,ONLY : JPIB, JPIM, JPRBT

    USE TPM_DISTR       ,ONLY : D, MYSETW, MYPROC, NPROC
    USE TPM_GEOMETRY    ,ONLY : G
    USE TPM_FFTC        ,ONLY : PLAN_DIR_FFT, EXECUTE_DIR_FFT, FFTTMPBUFFER
    USE MPL_MODULE      ,ONLY : MPL_BARRIER, MPL_ALL_MS_COMM, MPL_ABORT
    USE TPM_STATS, ONLY : GSTATS => GSTATS_NVTX
    USE ISO_C_BINDING, ONLY: C_NULL_PTR, C_PTR

    IMPLICIT NONE

    INTEGER(KIND=JPIM),INTENT(IN)  :: KFIELD
    REAL(KIND=JPRBT), INTENT(INOUT), POINTER :: PREEL_REAL(:)
    REAL(KIND=JPRBT), INTENT(OUT), POINTER :: PREEL_COMPLEX(:)
    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR
    TYPE(FTDIR_HANDLE) :: HFTDIR

    INTEGER(KIND=JPIM) :: KGL
    INTEGER(KIND=JPIB) :: IREQUIRED_SIZE
    REAL(KIND=JPRBT) :: DUMMY

    TYPE(C_PTR) :: LOCAL_FFTTMPBUFFER = C_NULL_PTR

    !     ------------------------------------------------------------------

    IREQUIRED_SIZE = PLAN_DIR_FFT(DUMMY, KFIELD, &
        & LOENS=G%NLOEN(D%NPTRLS(MYSETW):D%NPTRLS(MYSETW)+D%NDGL_FS-1), &
        & OFFSETS=D%NSTAGTF(1:D%NDGL_FS),LALLOCATE=.NOT. ALLOCATED(FFTTMPBUFFER))
    IF (ALLOCATED(FFTTMPBUFFER)) THEN
      IF (SIZE(FFTTMPBUFFER,KIND=JPIB) < IREQUIRED_SIZE) THEN
        WRITE(0,*), "FFT BUFFER SIZE EXPECTED: ", IREQUIRED_SIZE
        WRITE(0,*), "                ACTUAL:   ", SIZE(FFTTMPBUFFER,KIND=JPIB)
        CALL MPL_ABORT("too small FFT Buffer")
      ELSE
        !$ACC HOST_DATA USE_DEVICE(FFTTMPBUFFER)
        LOCAL_FFTTMPBUFFER = C_LOC(FFTTMPBUFFER)
        !$ACC END HOST_DATA
      ENDIF
    ELSE
      PRINT *, "FTDIR: REQUIRED SIZE", IREQUIRED_SIZE
    ENDIF

    PREEL_COMPLEX => PREEL_REAL

    !$ACC DATA PRESENT(PREEL_REAL,PREEL_COMPLEX)

    IF (LSYNC_TRANS) THEN
      CALL GSTATS(430,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(430,1)
    ENDIF
    CALL GSTATS(413,0)
    CALL EXECUTE_DIR_FFT(PREEL_REAL(:),PREEL_COMPLEX(:),KFIELD, &
        & LOENS=G%NLOEN(D%NPTRLS(MYSETW):D%NPTRLS(MYSETW)+D%NDGL_FS-1), &
        & OFFSETS=D%NSTAGTF(1:D%NDGL_FS),ALLOC=ALLOCATOR%PTR,BUF=LOCAL_FFTTMPBUFFER)

    IF (LSYNC_TRANS) THEN
      CALL GSTATS(433,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(433,1)
    ENDIF
    CALL GSTATS(413,1)

    !$ACC END DATA

    NULLIFY(PREEL_REAL)

    !     ------------------------------------------------------------------
  END SUBROUTINE FTDIR
END MODULE FTDIR_MOD
