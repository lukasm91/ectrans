SUBROUTINE DIR_TRANS(PSPVOR,PSPDIV,PSPSCALAR,&
& KPROMA,KVSETUV,KVSETSC,&
& PGP)

#include "tsmbkind.h"

REAL_B    ,OPTIONAL, INTENT(OUT) :: PSPVOR(:,:)
REAL_B    ,OPTIONAL, INTENT(OUT) :: PSPDIV(:,:)
REAL_B    ,OPTIONAL, INTENT(OUT) :: PSPSCALAR(:,:)
INTEGER_M ,OPTIONAL, INTENT(IN) :: KPROMA
INTEGER_M ,OPTIONAL, INTENT(IN) :: KVSETUV(:)
INTEGER_M ,OPTIONAL, INTENT(IN) :: KVSETSC(:)

REAL_B    ,INTENT(IN) :: PGP(:,:,:)

END SUBROUTINE DIR_TRANS
