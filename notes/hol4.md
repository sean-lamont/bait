# HOL4 environment notes

- toggle_simpset
  - Doesn't matter about diminish argument
  - Uses 'recreate_sset_at_parentage' not available in docs
    - Old version?
     
- Environment only takes string tactic arguments (good!), and has additional extra processing for tactic based models

- Initialisation: Takes library, imports/opens them
 
- When proving goal:
  - Toggles simpset of the theory the goal belongs to
  - Gets all valid premises by reference to a precomputed database
    - SML script to extract up to library
    - How to extend for e.g. GitHub repo, arbitrary library?
  - Sends 'g' + the goal to the environment
  - Sends tactic as a string
  - Processes response (list of pre-defined behaviours)

To adapt:

- Need retrieve_premises function
  - Should just be port of gen_fact_pool
- Need run_tactic
  - Port of query/step?
- Need get_hol4_theorem 
  - Config based on e.g. library name

Should be simple to port just the TacticZero version. To get better environment with new repos, libraries etc. not too sure
