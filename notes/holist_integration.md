# HOList

- Need to keep PB2 setup for environment, as that is what it takes as input
- Already have pb2 functionality for loading goals and generating premise sets, may as well reuse this 
- Get human proof logs from holist data and run generative training as with LeanDojo
- Make HOListEnv
  - Modify TacticApplication from proof_search_tree for run_tactics
  - get_premises can reuse pb2 premise sets
  - Generate docker container per instance with _enter_ then destroy with _exit_?
    - setup_prover from prover.py
- Sexpressions default encoding, not sure if we can use pretty printed..
  - Can parse from sexpression back?
  - Can use polished representation 
  - If goal includes hypotheses in pretty print, then can use that 
- Need to adapt processing of tactics (could just be restricted to e.g. fixed tac + premises currently)


# HOL4
- Should be much simpler, get_premises can come from gen_fact_pool
- Remove all reward handling for now
- run_tactic should be similar to step, takes a goal and a tactic and runs as in previous env
  - logic for timeouts etc. already there
- Enter/exit should just be similar to __init__ and close methods
