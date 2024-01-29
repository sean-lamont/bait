(* ========================================================================= *)
(* Area of a circle.                                                         *)
(* ========================================================================= *)

set_jrh_lexer;;
Pb_printer.set_file_tags ["Top100"; "circle.ml"];;

open Lib;;
open Fusion;;
open Parser;;
open Equal;;
open Bool;;
open Drule;;
open Tactics;;
open Simp;;
open Theorems;;
open Class;;
open Trivia;;
open Calc_num;;
open Realax;;
open Calc_int;;
open Realarith;;
open Reals;;
open Calc_rat;;
open Sets;;
open Cart;;
open Misc;;

open Vectors;;
open Determinants;;
open Topology;;
open Convex;;
open Integration;;
open Measure;;
open Transcendentals;;
open Realanalysis;;

(* ------------------------------------------------------------------------- *)
(* Circle area. Should maybe extend WLOG tactics for such scaling.           *)
(* ------------------------------------------------------------------------- *)

let AREA_UNIT_CBALL = prove
 (`measure(cball(vec 0:real^2,&1)) = pi`,
  REPEAT STRIP_TAC THEN
  MATCH_MP_TAC(INST_TYPE[`:1`,`:M`; `:2`,`:N`] FUBINI_SIMPLE_COMPACT) THEN
  EXISTS_TAC `1` THEN
  SIMP_TAC[DIMINDEX_1; DIMINDEX_2; ARITH; COMPACT_CBALL; SLICE_CBALL] THEN
  REWRITE_TAC[VEC_COMPONENT; DROPOUT_0; REAL_SUB_RZERO] THEN
  ONCE_REWRITE_TAC[COND_RAND] THEN REWRITE_TAC[MEASURE_EMPTY] THEN
  SUBGOAL_THEN `!t. abs(t) <= &1 <=> t IN real_interval[-- &1,&1]`
   (fun th -> REWRITE_TAC[th])
  THENL [REWRITE_TAC[IN_REAL_INTERVAL] THEN REAL_ARITH_TAC; ALL_TAC] THEN
  REWRITE_TAC[HAS_REAL_INTEGRAL_RESTRICT_UNIV; BALL_1] THEN
  MATCH_MP_TAC HAS_REAL_INTEGRAL_EQ THEN
  EXISTS_TAC `\t. &2 * sqrt(&1 - t pow 2)` THEN CONJ_TAC THENL
   [X_GEN_TAC `t:real` THEN SIMP_TAC[IN_REAL_INTERVAL; MEASURE_INTERVAL] THEN
    REWRITE_TAC[REAL_BOUNDS_LE; VECTOR_ADD_LID; VECTOR_SUB_LZERO] THEN
    DISCH_TAC THEN
    W(MP_TAC o PART_MATCH (lhs o rand) CONTENT_1 o rand o snd) THEN
    REWRITE_TAC[LIFT_DROP; DROP_NEG] THEN
    ANTS_TAC THENL [ALL_TAC; SIMP_TAC[REAL_POW_ONE] THEN REAL_ARITH_TAC] THEN
    MATCH_MP_TAC(REAL_ARITH `&0 <= x ==> --x <= x`) THEN
    ASM_SIMP_TAC[SQRT_POS_LE; REAL_SUB_LE; GSYM REAL_LE_SQUARE_ABS;
                 REAL_ABS_NUM];
    ALL_TAC] THEN
  MP_TAC(ISPECL
   [`\x.  asn(x) + x * sqrt(&1 - x pow 2)`;
    `\x. &2 * sqrt(&1 - x pow 2)`;
    `-- &1`; `&1`] REAL_FUNDAMENTAL_THEOREM_OF_CALCULUS_INTERIOR) THEN
  REWRITE_TAC[ASN_1; ASN_NEG_1] THEN CONV_TAC "100/circle.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  REWRITE_TAC[SQRT_0; REAL_MUL_RZERO; REAL_ADD_RID] THEN
  REWRITE_TAC[REAL_ARITH `x / &2 - --(x / &2) = x`] THEN
  DISCH_THEN MATCH_MP_TAC THEN CONJ_TAC THENL
   [MATCH_MP_TAC REAL_CONTINUOUS_ON_ADD THEN
    SIMP_TAC[REAL_CONTINUOUS_ON_ASN; IN_REAL_INTERVAL; REAL_BOUNDS_LE] THEN
    MATCH_MP_TAC REAL_CONTINUOUS_ON_MUL THEN
    REWRITE_TAC[REAL_CONTINUOUS_ON_ID] THEN
    GEN_REWRITE_TAC "100/circle.ml:LAND_CONV" LAND_CONV [GSYM o_DEF] THEN
    MATCH_MP_TAC REAL_CONTINUOUS_ON_COMPOSE THEN
    SIMP_TAC[REAL_CONTINUOUS_ON_SUB; REAL_CONTINUOUS_ON_POW;
             REAL_CONTINUOUS_ON_ID; REAL_CONTINUOUS_ON_CONST] THEN
    MATCH_MP_TAC REAL_CONTINUOUS_ON_SQRT THEN
    REWRITE_TAC[FORALL_IN_IMAGE; IN_REAL_INTERVAL] THEN
    REWRITE_TAC[REAL_ARITH `&0 <= &1 - x <=> x <= &1 pow 2`] THEN
    REWRITE_TAC[GSYM REAL_LE_SQUARE_ABS; REAL_ABS_NUM] THEN
    REAL_ARITH_TAC;
    REWRITE_TAC[IN_REAL_INTERVAL; REAL_BOUNDS_LT] THEN REPEAT STRIP_TAC THEN
    REAL_DIFF_TAC THEN
    CONV_TAC "100/circle.ml:NUM_REDUCE_CONV" NUM_REDUCE_CONV THEN
    REWRITE_TAC[REAL_MUL_LID; REAL_POW_1; REAL_MUL_RID] THEN
    REWRITE_TAC[REAL_SUB_LZERO; REAL_MUL_RNEG; REAL_INV_MUL] THEN
    ASM_REWRITE_TAC[REAL_SUB_LT; ABS_SQUARE_LT_1] THEN
    MATCH_MP_TAC(REAL_FIELD
     `s pow 2 = &1 - x pow 2 /\ x pow 2 < &1
      ==> (inv s + x * --(&2 * x) * inv (&2) * inv s + s) = &2 * s`) THEN
    ASM_SIMP_TAC[ABS_SQUARE_LT_1; SQRT_POW_2; REAL_SUB_LE; REAL_LT_IMP_LE]]);;

let AREA_CBALL = prove
 (`!z:real^2 r. &0 <= r ==> measure(cball(z,r)) = pi * r pow 2`,
  REPEAT STRIP_TAC THEN ASM_CASES_TAC `r = &0` THENL
   [ASM_SIMP_TAC[CBALL_SING; REAL_POW_2; REAL_MUL_RZERO] THEN
    MATCH_MP_TAC MEASURE_UNIQUE THEN
    REWRITE_TAC[HAS_MEASURE_0; NEGLIGIBLE_SING];
    ALL_TAC] THEN
  SUBGOAL_THEN `&0 < r` ASSUME_TAC THENL [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
  MP_TAC(ISPECL [`cball(vec 0:real^2,&1)`; `r:real`; `z:real^2`; `pi`]
        HAS_MEASURE_AFFINITY) THEN
  REWRITE_TAC[HAS_MEASURE_MEASURABLE_MEASURE; MEASURABLE_CBALL;
              AREA_UNIT_CBALL] THEN
  ASM_REWRITE_TAC[real_abs; DIMINDEX_2] THEN
  DISCH_THEN(MP_TAC o CONJUNCT2) THEN
  GEN_REWRITE_TAC "100/circle.ml:(LAND_CONV o ONCE_DEPTH_CONV)" (LAND_CONV o ONCE_DEPTH_CONV) [REAL_MUL_SYM] THEN
  DISCH_THEN(SUBST1_TAC o SYM) THEN AP_TERM_TAC THEN
  MATCH_MP_TAC SUBSET_ANTISYM THEN REWRITE_TAC[SUBSET; FORALL_IN_IMAGE] THEN
  REWRITE_TAC[IN_CBALL_0; IN_IMAGE] THEN REWRITE_TAC[IN_CBALL] THEN
  REWRITE_TAC[NORM_ARITH `dist(z,a + z) = norm a`; NORM_MUL] THEN
  ONCE_REWRITE_TAC[REAL_ARITH `abs r * x <= r <=> abs r * x <= r * &1`] THEN
  ASM_SIMP_TAC[real_abs; REAL_LE_LMUL; dist] THEN X_GEN_TAC `w:real^2` THEN
  DISCH_TAC THEN EXISTS_TAC `inv(r) % (w - z):real^2` THEN
  ASM_SIMP_TAC[VECTOR_MUL_ASSOC; REAL_MUL_RINV] THEN
  CONJ_TAC THENL [NORM_ARITH_TAC; ALL_TAC] THEN
  REWRITE_TAC[NORM_MUL; REAL_ABS_INV] THEN ASM_REWRITE_TAC[real_abs] THEN
  ONCE_REWRITE_TAC[REAL_MUL_SYM] THEN
  ASM_SIMP_TAC[GSYM real_div; REAL_LE_LDIV_EQ; REAL_MUL_LID] THEN
  ONCE_REWRITE_TAC[NORM_SUB] THEN ASM_REWRITE_TAC[]);;

let AREA_BALL = prove
 (`!z:real^2 r. &0 <= r ==> measure(ball(z,r)) = pi * r pow 2`,
  SIMP_TAC[GSYM INTERIOR_CBALL; GSYM AREA_CBALL] THEN
  REPEAT STRIP_TAC THEN MATCH_MP_TAC MEASURE_INTERIOR THEN
  SIMP_TAC[BOUNDED_CBALL; NEGLIGIBLE_CONVEX_FRONTIER; CONVEX_CBALL]);;

(* ------------------------------------------------------------------------- *)
(* Volume of a ball too, just for fun.                                       *)
(* ------------------------------------------------------------------------- *)

let VOLUME_CBALL = prove
 (`!z:real^3 r. &0 <= r ==> measure(cball(z,r)) = &4 / &3 * pi * r pow 3`,
  GEOM_ORIGIN_TAC `z:real^3` THEN REPEAT STRIP_TAC THEN
  MATCH_MP_TAC(INST_TYPE[`:2`,`:M`; `:3`,`:N`] FUBINI_SIMPLE_COMPACT) THEN
  EXISTS_TAC `1` THEN
  SIMP_TAC[DIMINDEX_2; DIMINDEX_3; ARITH; COMPACT_CBALL; SLICE_CBALL] THEN
  REWRITE_TAC[VEC_COMPONENT; DROPOUT_0; REAL_SUB_RZERO] THEN
  ONCE_REWRITE_TAC[COND_RAND] THEN REWRITE_TAC[MEASURE_EMPTY] THEN
  SUBGOAL_THEN `!t. abs(t) <= r <=> t IN real_interval[--r,r]`
   (fun th -> REWRITE_TAC[th])
  THENL [REWRITE_TAC[IN_REAL_INTERVAL] THEN REAL_ARITH_TAC; ALL_TAC] THEN
  REWRITE_TAC[HAS_REAL_INTEGRAL_RESTRICT_UNIV] THEN
  MATCH_MP_TAC HAS_REAL_INTEGRAL_EQ THEN
  EXISTS_TAC `\t. pi * (r pow 2 - t pow 2)` THEN CONJ_TAC THENL
   [X_GEN_TAC `t:real` THEN REWRITE_TAC[IN_REAL_INTERVAL; REAL_BOUNDS_LE] THEN
    SIMP_TAC[AREA_CBALL; SQRT_POS_LE; REAL_SUB_LE; GSYM REAL_LE_SQUARE_ABS;
             SQRT_POW_2; REAL_ARITH `abs x <= r ==> abs x <= abs r`];
    ALL_TAC] THEN
  MP_TAC(ISPECL
   [`\t. pi * (r pow 2 * t - &1 / &3 * t pow 3)`;
    `\t. pi * (r pow 2 - t pow 2)`;
    `--r:real`; `r:real`] REAL_FUNDAMENTAL_THEOREM_OF_CALCULUS) THEN
  REWRITE_TAC[] THEN ANTS_TAC THENL
   [CONJ_TAC THENL [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
    REPEAT STRIP_TAC THEN REAL_DIFF_TAC THEN
    CONV_TAC "100/circle.ml:NUM_REDUCE_CONV" NUM_REDUCE_CONV THEN CONV_TAC "100/circle.ml:REAL_RING" REAL_RING;
    MATCH_MP_TAC EQ_IMP THEN AP_THM_TAC THEN AP_TERM_TAC THEN
    CONV_TAC "100/circle.ml:REAL_RING" REAL_RING]);;

let VOLUME_BALL = prove
 (`!z:real^3 r. &0 <= r ==> measure(ball(z,r)) =  &4 / &3 * pi * r pow 3`,
  SIMP_TAC[GSYM INTERIOR_CBALL; GSYM VOLUME_CBALL] THEN
  REPEAT STRIP_TAC THEN MATCH_MP_TAC MEASURE_INTERIOR THEN
  SIMP_TAC[BOUNDED_CBALL; NEGLIGIBLE_CONVEX_FRONTIER; CONVEX_CBALL]);;
Pb_printer.clear_file_tags();;