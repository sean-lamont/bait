## Refactor notes

- Retriever
  - All environments have some notion of retrieval
      - HOList has database with position/fingerprints
      - HOL4 has int as position in library
      - Lean has position as well
  - In all cases, we need to retrieve allowed premises, regardless of tactic model
  - Depending on tactic model, premises are chosen before or after tactic
  - Seems like an essential part of the system, possibly make a separate retriever module (part of environment?)
  - Retrieved premises fed as input into tactic model. 
    - ReProver will have retrieval model as first step of tac_model
    - TacticZero/HOList agent will have tactic first, then ranking 