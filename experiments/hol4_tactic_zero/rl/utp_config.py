import json
import re
import random
import pickle
random.seed(0)

HOLPATH = "/home/wu099/temp/HOL/bin/hol --maxheap=256"
CACHE_PATH = "/scratch1/wu099/temp/HOL_cache.pkl"

try: 
    with open("dict.json") as f:
        dictionary = json.load(f)
except:
    dictionary = {}

try: 
    with open("provables.json") as f:
        provables = json.load(f)
except:
    provables = []

provables = [t[0] for t in provables]

try:
    with open("bigger_new_facts.json") as f:
        new_facts = json.load(f)
except:
    new_facts = {}

try: 
    with open(CACHE_PATH, "rb") as f:
        HOL_cache = pickle.load(f)
except:
    HOL_cache = {}


MAX_LEN = 128
MAX_ASSUMPTIONS = 3
MAX_CONTEXTS = 8
PRINT_EXCEPTION = False
UNEXPECTED_REWARD = -1000
# TARGET_THEORIES = ["probability", "martingale", "lebesgue", "borel", "real_borel", "sigma_algebra","util_prob", "fcp", "indexedLists", "rich_list", "list", "pred_set","numpair", "basicSize", "numeral", "arithmetic", "prim_rec", "num","marker", "bool", "min", "normalForms", "relation", "sum", "pair", "graph_benchmarks","while", "bit", "logroot", "transc", "powser", "lim", "seq", "nets","metric", "real", "realax", "hreal", "hrat", "quotient_sum", "quotient","res_quan", "product", "iterate", "cardinal", "wellorder","set_relation", "derivative", "real_topology"]

# TARGET_THEORIES = ["probability", "martingale", "lebesgue", "borel", "real_borel", "sigma_algebra","util_prob", "fcp", "indexedLists", "rich_list", "list", "pred_set","numpair", "basicSize", "numeral", "prim_rec", "num","marker", "normalForms", "relation", "sum", "pair", "graph_benchmarks","while", "bit", "logroot", "transc", "powser", "lim", "seq", "nets","metric", "real", "realax", "hreal", "hrat", "quotient_sum", "quotient","res_quan", "product", "iterate", "cardinal", "wellorder","set_relation", "derivative", "real_topology"]

# TARGET_THEORIES = ["real_topology"] # ["iterate"] #["pred_set"] # ["real"] # 
# TARGET_THEORIES = ["pred_set"] # ["real"] #
TARGET_THEORIES = ["list"] # ["real"] # 

EXCLUDED_THEORIES = ["min"] #["min", "bool"]
# MODE = "train"
CONTINUE = False
ALLOW_NEW_FACTS = False #True
MORE_TACTICS = True

# with open("polished_def_dict.json") as f:
#     defs = json.load(f)
    # print(list(dictionary.keys()))
# with open("polished_thm_dict_sorted.json") as f:
#     pthms = json.load(f)
with open("../../../data/hol4/data_v2/data/include_probability.json") as f:
    database = json.load(f)

# if CONTINUE:
#     with open("fact_pool.json") as f:
#         fact_pool = json.load(f)
# else:
#     fact_pool = list(defs.keys())

reverse_database = {(value[0], value[1]) : key for key, value in database.items()}

PROVABLES = [value[4] for key, value in database.items() if value[0] == "list" and value[1] in provables]

if not MORE_TACTICS:
    thms_tactic = ["simp", "fs", "metis_tac"]
    thm_tactic = ["irule"]
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac"]
else:
    thms_tactic = ["simp", "fs", "metis_tac", "rw", "prove_tac", "asm_rewrite_tac","once_rewrite_tac"]
    thm_tactic = ["irule", "drule", "imp_res_tac", "match_mp_tac"] # , "assume_tac"
    term_tactic = ["Induct_on"]
    no_arg_tactic = ["strip_tac", "eq_tac", "decide_tac", "first_x_assum match_mp_tac"] # , "CCONTR_TAC"

    # thms_tactic = ["simp", "fs", "metis_tac", "rw", "prove_tac", "asm_rewrite_tac","once_rewrite_tac", "ASM_MESON_TAC", "MESON_TAC"]
    # thm_tactic = ["irule", "drule", "imp_res_tac", "match_mp_tac"] # , "assume_tac"
    # term_tactic = ["Induct_on"]
    # # no_arg_tactic = ["strip_tac", "eq_tac", "decide_tac", "first_x_assum match_mp_tac", "REAL_ARITH_TAC", "ARITH_TAC", "SET_TAC[]", "ASM_SET_TAC[]"] # , "CCONTR_TAC"
    # no_arg_tactic = ["strip_tac", "eq_tac", "decide_tac", "first_x_assum match_mp_tac", "REAL_ARITH_TAC", "ARITH_TAC"] # , "CCONTR_TAC"

    # thms_tactic = ["simp", "fs", "metis_tac", "rw"]
    # thm_tactic = [] #["irule", "drule"] 
    # term_tactic = ["Induct_on"]
    # no_arg_tactic = ["strip_tac", "EQ_TAC", "simp[]", "rw[]", "metis_tac[]", "fs[]"]

tactic_pool = thms_tactic + thm_tactic + term_tactic + no_arg_tactic

# with open("thm_dict_sorted.json") as f:
#     thms = json.load(f)
#     # print(list(dictionary.keys()))

# original = list(thms.keys())

GOALS = [value[4] for key, value in database.items() if value[3] == "thm" and value[0] in TARGET_THEORIES]
plain_database = {value[4] : [value[0], value[1], value[2], value[3], key] for key, value in database.items()}
# GOALS = [t for t in database if database[t][0] in TARGET_THEORIES]
# TEST_GOALS = [GOALS[5]]
SMALL = ["∀c l. EXISTS (λx. c) l ⇔ l ≠ [] ∧ c",
             "REVERSE l = [] ⇔ l = []",
             "∀l. l = [] ∨ ∃h t. l = h::t",
             "∀l1 l2 l3. l1 ++ (l2 ++ l3) = l1 ++ l2 ++ l3",
             "∀M M' v f. M = M' ∧ (M' = [] ⇒ v = v') ∧ (∀a0 a1. M' = a0::a1 ⇒ f a0 a1 = f' a0 a1) ⇒ list_CASE M v f = list_CASE M' v' f'",
             "l1 ++ l2 = [e] ⇔ l1 = [e] ∧ l2 = [] ∨ l1 = [] ∧ l2 = [e]",
             "LAST (h::t) = if t = [] then h else LAST t",
             "0 = LENGTH l ⇔ l = []",
             "¬SHORTLEX R l []",
             "list_CASE x v f = v' ⇔ x = [] ∧ v = v' ∨ ∃a l. x = a::l ∧ f a l = v'"]

TYPED_SMALL = ["∀(c :bool) (l :α list). EXISTS (λ(x :α). c) l ⇔ l ≠ ([] :α list) ∧ c",
               "WF ($<< :α -> α -> bool) ⇔ ¬∃(s :num -> α). ∀(n :num). s (SUC n) ≪ s n",
               "REVERSE (l :α list) = ([] :α list) ⇔ l = ([] :α list)",
               "∀(l :α list). l = ([] :α list) ∨ ∃(h :α) (t :α list). l = h::t",
               "∀(l1 :α list) (l2 :α list) (l3 :α list). l1 ++ (l2 ++ l3) = l1 ++ l2 ++ l3",
               "∀(M :α list) (M' :α list) (v :β) (f :α -> α list -> β). M = M' ∧ (M' = ([] :α list) ⇒ v = (v' :β)) ∧ (∀(a0 :α) (a1 :α list). M' = a0::a1 ⇒ f a0 a1 = (f' :α -> α list -> β) a0 a1) ⇒ (list_CASE M v f :β) = (list_CASE M' v' f' :β)",
               "(l1 :α list) ++ (l2 :α list) = [(e :α)] ⇔ l1 = [e] ∧ l2 = ([] :α list) ∨ l1 = ([] :α list) ∧ l2 = [e]",
               "LAST ((h :α)::(t :α list)) = if t = ([] :α list) then h else LAST t",
               "(0 :num) = LENGTH (l :α list) ⇔ l = ([] :α list)",
               "¬SHORTLEX (R :α -> α -> bool) (l :α list) ([] :α list)",
               "(list_CASE (x :α list) (v :β) (f :α -> α list -> β) :β) = (v' :β) ⇔ x = ([] :α list) ∧ v = v' ∨ ∃(a :α) (l :α list). x = a::l ∧ f a l = v'"]
# LARGER = PROVABLES + ["∀l. ZIP (UNZIP l) = l",
#                       "ZIP ([],[]) = [] ∧ ∀x1 l1 x2 l2. ZIP (x1::l1,x2::l2) = (x1,x2)::ZIP (l1,l2)",
#                       "∀l1 l2. LENGTH l1 = LENGTH l2 ⇒ UNZIP (ZIP (l1,l2)) = (l1,l2)",
#                       "UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))",
#                       "∀f n. TL (GENLIST f (SUC n)) = GENLIST (f ∘ SUC) n",
#                       "∀m n. TAKE n (TAKE m l) = TAKE (MIN n m) l",
#                       "∀l n. LENGTH l ≤ n ⇒ TAKE n l = l",
#                       "TAKE n (GENLIST f m) = GENLIST f (MIN n m)",
#                       "∀l. ALL_DISTINCT l ⇔ ∀x. MEM x l ⇒ FILTER ($= x) l = [x]",
#                       "∀xs. ALL_DISTINCT (FLAT (REVERSE xs)) ⇔ ALL_DISTINCT (FLAT xs)",
#                       "∀l. ¬NULL l ⇒ HD l::TL l = l",
#                       "∀l1 x l2. l1 ++ SNOC x l2 = SNOC x (l1 ++ l2)",
#                       "∀l n. LENGTH l ≤ n ⇒ DROP n l = []",
#                       "∀ls n. DROP n ls = [] ⇔ n ≥ LENGTH ls",
#                       "∀f n x. x < n ⇒ EL x (GENLIST f n) = f x",
#                       "∀n. EL n l = if n = 0 then HD l else EL (PRE n) (TL l)",
#                       "EL 0 = HD ∧ EL (SUC n) (l::ls) = EL n ls",
#                       "∀n x l. x < n ⇒ EL x (TAKE n l) = EL x l",
#                       "∀R l1 l2. LIST_REL R l1 l2 ⇒ LIST_REL R (REVERSE l1) (REVERSE l2)",
#                       "(∀x. MEM x ls ⇒ R x x) ⇒ LIST_REL R ls ls"]

# PROVABLES = ["REVERSE [] = [] ∧ ∀x l. REVERSE (x::l) = SNOC x (REVERSE l)",
#               "∀l1 l2. REVERSE (l1 ++ l2) = REVERSE l2 ++ REVERSE l1",
#               "REVERSE l = [] ⇔ l = []",
#               "∀f x l. MAP f (SNOC x l) = SNOC (f x) (MAP f l)",
#               "UNZIP [] = ([],[]) ∧ UNZIP ((x,y)::t) = (let (L1,L2) = UNZIP t in (x::L1,y::L2))",
#               "∀h1 h2. h1 ≠ h2 ⇒ ∀l1 l2. h1::l1 ≠ h2::l2",
#               "∀n f. NULL (GENLIST f n) ⇔ n = 0",
#               "¬LLEX R l []",
#               "∀P l. EVERY P l ⇔ ¬EXISTS (λx. ¬P x) l",
#               "LIST_BIND [x] f = f x"]

# MAYBE = ["LENGTH (FST ps) = LENGTH (SND ps) ∧ MEM p (ZIP ps) ⇒ MEM (FST p) (FST ps) ∧ MEM (SND p) (SND ps)",
#          "∀l f x. MEM x (MAP f l) ⇔ ∃y. x = f y ∧ MEM y l",
#          "∀x ls n. MEM x (DROP n ls) ⇔ ∃m. m + n < LENGTH ls ∧ x = EL (m + n) ls",
#          "∀ls n. ALL_DISTINCT ls ⇒ ALL_DISTINCT (DROP n ls)",
#          "BIGUNION (IMAGE f (set ls)) ⊆ s ⇔ ∀x. MEM x ls ⇒ f x ⊆ s",
#          "DATATYPE (list [] CONS)",
#          "∀n l. n < LENGTH l ⇒ ∀x. EL n (SNOC x l) = EL n l",
#          "∀P l1 l2. EXISTS P (l1 ++ l2) ⇔ EXISTS P l1 ∨ EXISTS P l2",
#          "∀x l. FRONT (SNOC x l) = l",
#          "∀f l1 l2. INJ f (set l1 ∪ set l2) 𝕌(:β) ⇒ (MAP f l1 = MAP f l2 ⇔ l1 = l2)"]

TEST_GOALS = PROVABLES
# TEST_GOALS = SMALL

random.shuffle(TEST_GOALS)
random.shuffle(GOALS)
TEST = GOALS[:(len(GOALS)//4)]
TRAIN = GOALS[(len(GOALS)//4):]
