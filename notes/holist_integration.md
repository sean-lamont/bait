# HOList

- Need to keep PB2 setup for environment, as that is what it takes as input
- Already have pb2 functionality for loading goals and generating premise sets, may as well reuse this 
- Get human proof logs from holist data and run generative training as with LeanDojo
- Make HOListEnv
  - Modify TacticApplication from proof_search_tree for run_tactics, with error handling etc
  - get_premises can reuse pb2 premise sets
  - Generate docker container per instance with _enter_ then destroy with _exit_?
    - setup_prover from prover.py
- Sexpressions default encoding, not sure if we can use pretty printed..
  - Can parse from sexpression back?
  - Can use polished representation 
  - If goal includes hypotheses in pretty print, then can use that 
- Need to adapt processing of tactics (could just be restricted to e.g. fixed tac + premises currently)
  - ApplyTacticRequest is sent to environment. Takes tactic string, timeout, and goal of type Theorem
    - Keep dictionary mapping from goal string to HOList Theorem Type
- Need to look at proof_checker_lib and adapt so we can verify proofs in HOL Light
- Need to check additional processing on theorem fingerprints etc, i.e. how to generate parameters 
- Holparampredictor to become an instance of tac_model (class to map from goal/premises to tactic)
- ProofLogToExamples reimplement with new SearchResult, then can be verified with O'Caml script



- Obvious that s-expressions are fundamental to the HOList implementation... 
  - E.g. theorem fingerprinting, proof replay and verification, registering theorems, etc...
  - Therefore tactic interface takes:
    - No parameter (fine)
    - Theorem/list of theorems (just formatted as tac [t1; t2; t3; ], where ti are the fingerprints).
      - This is fine as we can get a generative model to just identify the name of the theorem, which we then convert 
      to a fingerprint in environment processing
    - Term (takes an s-expression, currently not fine)
  - Pretty printed can be found from environment interaction though, and s-expressions are extremely verbose..
  - Can't add new/arbitrary assumptions unless from a pre-existing theorem/definition in the database
    - Can't register theorem/goals with hypotheses, so can't add new assumptions
  - Overall quite restrictive
    - Human proof data is only in s-expressions
    - Can't use any new assumptions 
    - Tactics are only from a fixed list, and can only take other registered theorems or s-expression terms as input
  - Can still use ReProver style except:
    - Only a small number of tactics
    - Post-processing to convert tactic name to fingerprint
    - Human data is in s-expression format, very significant increase over PP 
      - Environment can give us pretty printed
      - Could try find out how to regenerate human proof data with pretty printed expressions?
        - Even then, terms still need to be in s-expression format, so would be restricted to theorem only parameters
    - If we can get PP human labelled data, then would be a better platform.
     Still restricted by the number and type of tactics, but should be much more concise than s-expressions

## Status
- Environment gives PP expressions for both goals and theorems
- Can get these for human proof logs as well, although it is some effort to get this running nicely 
- Core and complex seem to come built in, and covered in the Makefile. 
- Flyspeck is covered in `https://github.com/flyspeck/flyspeck/`
  - Can look at the original theorem database to collect which theorems are necessary, however then have to build 
  all of these within the docker container

# HOL4
- Should be much simpler, get_premises can come from gen_fact_pool
- Remove all reward handling for now
- run_tactic should be similar to step, takes a goal and a tactic and runs as in previous env
  - logic for timeouts etc. already there
- Enter/exit should just be similar to __init__ and close methods
- Polished is much less verbose than HOList s-expressions, can add PP as well 

# Misc
- process_traces should have a method for determining premises so they can be labelled 
  - Needs to define how to process traces for different model types (e.g. generative will be different to embedding based)
    - Create datasets for model to train on 