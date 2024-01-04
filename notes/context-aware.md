# Idea: Context aware proof synthesis

- Initial state is the goal
- Context is additional input to the state, concatenated with the goal
  - Relevant premises (reprover)
  - Context from theory file (proof repair)
  - Previous failed attempts (proof repair) (hard to train)
  - Previous successful examples (diffusion paper) (hard to train)
  - NLP annotations (e.g. subgoals from diffusion paper) (hard to train)
- Have model to score a state with the context
- Agent has the option of:
  - Taking a tactic to run in the ITP
  - Updating the context using any submodule (e.g. retrieve premises, get failures, get similar example, add annotation)
- New child made, either with sub-goal from ITP or with updated context

Simpler:
- Have ReProver + Theory file context