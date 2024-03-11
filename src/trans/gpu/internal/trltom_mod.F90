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

MODULE TRLTOM_MOD
  USE BUFFERED_ALLOCATOR_MOD
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: TRLTOM_CUDAAWARE, PREPARE_TRLTOM, TRLTOM_HANDLE

  TYPE TRLTOM_HANDLE
    TYPE(ALLOCATION_RESERVATION_HANDLE) :: HPFBUF
  END TYPE
CONTAINS
  FUNCTION PREPARE_TRLTOM(ALLOCATOR, KF_FS) RESULT(HTRLTOM)
    USE PARKIND_ECTRANS, ONLY: JPIM, JPRBT, JPIB
    USE TPM_DISTR, ONLY: D

    IMPLICIT NONE

    TYPE(BUFFERED_ALLOCATOR), INTENT(INOUT) :: ALLOCATOR
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_FS
    TYPE(TRLTOM_HANDLE) :: HTRLTOM

    REAL(KIND=JPRBT) :: DUMMY

    HTRLTOM%HPFBUF = RESERVE(ALLOCATOR, 2_JPIB*D%NLENGT1B*KF_FS*SIZEOF(DUMMY))
  END FUNCTION

  SUBROUTINE TRLTOM_CUDAAWARE(ALLOCATOR,HTRLTOM,PFBUF_IN,PFBUF,KF_FS)
    !**** *TRLTOM * - transposition in Fourierspace

    !     Purpose.
    !     --------
    !              Transpose Fourier coefficients from partitioning
    !              over latitudes to partitioning over wave numbers
    !              This is done between inverse Legendre Transform
    !              and inverse FFT.
    !              This is the inverse routine of TRMTOL.

    !**   Interface.
    !     ----------
    !        *CALL* *TRLTOM(...)*

    !        Explicit arguments : PFBUF  - Fourier coefficient buffer. It is
    !        --------------------          used for both input and output.

    !                             KF_FS - Number of fields communicated

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
    !        Original : 95-10-01
    !        Modified : 97-06-18 G. Mozdzynski - control MPI mailbox use
    !                                            (NCOMBFLEN) for nphase.eq.1
    !        Modified : 99-05-28  D.Salmond - Optimise copies.
    !        Modified : 00-02-02  M.Hamrud  - Remove NPHASE
    !        D.Salmond : 01-11-23 LIMP_NOOLAP Option for non-overlapping message
    !                             passing and buffer packing
    !        G.Mozdzynski: 08-01-01 Cleanup
    !        Y.Seity   : 07-08-30 Add barrier synchronisation under LSYNC_TRANS
    !     ------------------------------------------------------------------

    USE PARKIND_ECTRANS  ,ONLY : JPIM     ,JPRBT, JPIB
    USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK, JPHOOK
    USE MPL_MODULE  ,ONLY : MPL_ALLTOALLV, MPL_BARRIER, MPL_ALL_MS_COMM, MPL_MYRANK, MPL_ABORT
    USE TPM_DISTR       ,ONLY : D, NPRTRW, NPROC, MYPROC, MYSETW
    USE TPM_GEN         ,ONLY : LSYNC_TRANS
    USE MPI
    USE TPM_STATS, ONLY : GSTATS => GSTATS_NVTX

    IMPLICIT NONE

    INTEGER(KIND=JPIM) ,INTENT(IN)  :: KF_FS
    REAL(KIND=JPRBT)   ,INTENT(OUT), POINTER  :: PFBUF(:)
    REAL(KIND=JPRBT)   ,INTENT(INOUT), POINTER :: PFBUF_IN(:)

    INTEGER(KIND=JPIM) :: ILENS(NPRTRW),ILENR(NPRTRW)
    INTEGER(KIND=JPIB) :: IOFFS(NPRTRW),IOFFR(NPRTRW)
    INTEGER(KIND=JPIM) :: J, FROM_SEND, TO_SEND, FROM_RECV, TO_RECV, IRANK
    INTEGER(KIND=JPIB) :: ILEN, ISTA, IEND
    REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
    INTEGER(KIND=JPIM) :: IERROR

    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR
    TYPE(TRLTOM_HANDLE), INTENT(IN) :: HTRLTOM

    #ifdef PARKINDTRANS_SINGLE
    #define TRLTOM_DTYPE MPI_REAL
    #else
    #define TRLTOM_DTYPE MPI_DOUBLE_PRECISION
    #endif

    IF (LHOOK) CALL DR_HOOK('TRLTOM_CUDAAWARE',0,ZHOOK_HANDLE)

    CALL ASSIGN_PTR(PFBUF, GET_ALLOCATION(ALLOCATOR, HTRLTOM%HPFBUF),&
        & 1_JPIB, 2_JPIB*D%NLENGT1B*KF_FS*SIZEOF(PFBUF(1)))

    !$ACC DATA PRESENT(PFBUF,PFBUF_IN)

    IF(NPROC > 1) THEN
      DO J=1,NPRTRW
        ILENS(J) = 2_JPIB*D%NLTSGTB(J)*KF_FS
        IOFFS(J) = 2_JPIB*D%NSTAGT0B(J)*KF_FS
        ILENR(J) = 2_JPIB*D%NLTSFTB(J)*KF_FS
        IOFFR(J) = 2_JPIB*D%NSTAGT1B(J)*KF_FS
      ENDDO

      CALL GSTATS(806,0)

      ! copy to self workaround
      IRANK = MPL_MYRANK(MPL_ALL_MS_COMM)
      IF (ILENS(IRANK) .ne. ILENR(IRANK)) THEN
          PRINT *, "ERROR", ILENS(IRANK), ILENR(IRANK)
          stop 1
      ENDIF
      IF (ILENS(IRANK) > 0) THEN
          FROM_SEND = IOFFS(IRANK) + 1
          TO_SEND = FROM_SEND + ILENS(IRANK) - 1
          FROM_RECV = IOFFR(IRANK) + 1
          TO_RECV = FROM_RECV + ILENR(IRANK) - 1
          !$ACC KERNELS ASYNC(1)
          PFBUF(FROM_RECV:TO_RECV) = PFBUF_IN(FROM_SEND:TO_SEND)
          !$ACC END KERNELS
          ILENS(IRANK) = 0
          ILENR(IRANK) = 0
      ENDIF

      IF (LSYNC_TRANS) THEN
        CALL GSTATS(430,0)
        CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
        CALL GSTATS(430,1)
      ENDIF
      CALL GSTATS(415,0)
      IF (ANY(INT(IOFFR,KIND=JPIM) /= INT(IOFFR,KIND=JPIB))) CALL MPL_ABORT("Overflow in trltom")
      IF (ANY(INT(IOFFS,KIND=JPIM) /= INT(IOFFS,KIND=JPIB))) CALL MPL_ABORT("Overflow in trltom")
      !$ACC HOST_DATA USE_DEVICE(PFBUF_IN, PFBUF)
      CALL MPI_ALLTOALLV(PFBUF_IN,ILENS,INT(IOFFS,KIND=JPIM),TRLTOM_DTYPE,&
       & PFBUF,ILENR,INT(IOFFR,KIND=JPIM), TRLTOM_DTYPE, &
       & MPL_ALL_MS_COMM,IERROR)
      !$ACC END HOST_DATA
      IF (LSYNC_TRANS) THEN
        CALL GSTATS(435,0)
        CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
        CALL GSTATS(435,1)
      ENDIF
      CALL GSTATS(415,1)

      !$ACC WAIT(1)
      CALL GSTATS(806,1)
    ELSE
      ILEN = 2_JPIB*D%NLTSGTB(MYSETW)*KF_FS
      ISTA = 2_JPIB*D%NSTAGT1B(MYSETW)*KF_FS+1
      IEND = ISTA + ILEN - 1
      CALL GSTATS(1607,0)
      !$ACC KERNELS ASYNC(1) DEFAULT(NONE)
      PFBUF(ISTA:IEND) = PFBUF_IN(ISTA:IEND)
      !$ACC END KERNELS
      CALL GSTATS(1607,1)
    ENDIF

    !$ACC END DATA

    IF (LHOOK) CALL DR_HOOK('TRLTOM_CUDAAWARE',1,ZHOOK_HANDLE)
    !     ------------------------------------------------------------------
  END SUBROUTINE TRLTOM_CUDAAWARE
END MODULE TRLTOM_MOD
