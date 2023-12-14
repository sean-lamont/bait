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
- ProofLogToExamples reimplement with new SearchResult


# HOL4
- Should be much simpler, get_premises can come from gen_fact_pool
- Remove all reward handling for now
- run_tactic should be similar to step, takes a goal and a tactic and runs as in previous env
  - logic for timeouts etc. already there
- Enter/exit should just be similar to __init__ and close methods

# Misc
- process_traces should have a method for determining premises so they can be labelled 
  - Needs to define how to process traces for different model types (e.g. generative will be different to embedding based)
    - Create datasets for model to train on 