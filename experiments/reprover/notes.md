# Notes

## Proof search

- TacticState returns only one new state after a tactic, with possibly multiple goals contained within
- The state given to the model is the pretty printed state with all goals (separated by a newline, with 
all assumptions above each node)
  - Small number of cases where "n goals: tactic_state" appears in state_before (~5k vs 30k where the states are generated)
  - All other cases seem to just be a single goal as the  before_state context


## Representation
Aim:
- Minimise redundancy (if a possible proof exists by combining paths, we want to find it)
  - Done by separating subgoals, and allowing multiple parents per goal (DAG rather than tree)
    - From observing proof traces, many similar tactics lead to previous goals
  - Only limited by cycles, which could remove some possibly proofs (not worth the trouble, many algorithms break when cycles exist)
    - Cycles detected by constant monitoring/updating of ancestor nodes (lightweight, only tracks the strings as nodes are uniquely hashed by the goal)
- Have rich training data 
  - Done by maintaining all tactic applications and the outcome, even if invalid
  - Isolates tactic to a single goal, possibly makes it easier for the model to learn good state -> tactic mapping
  - Include time spent per tactic, model logprobs, visit count
- Enable multiple search algos
  - Fringe/Original LeanDojo state doesn't allow for HTPS (as subgoals aren't separated) or UpDown 
  - Including context (what other goals need to be proven to prove the original goal), allows for e.g. UpDown style of approach
    - Context now includes all observed paths to a goal (to minimise redundancy). Selection algorithm can then take the best 
    option out of all paths (e.g. one context with x <= y, one with x < y, x <= y weaker assumption so should be easier to prove for the same goal)
  - (CASPER new name? Context Aware Search Procedure for Elaborated Reasoning)

### Separating subgoals
- How to implement separate subgoal search tree:
  - Keep LeanDojo module the same
  - Will need to change proof search and search tree classes. 
  - As LeanDojo takes entire tactic_state as input, will need to implement a tactic to select a particular subgoal given a full tactic state 
  - Note; Tactic application in lean is to the first goal of a tactic state. 
    - How to select a particular goal? Something like rotate_left in https://github.com/leanprover-community/lean/blob/master/library/init/meta/tactic.lean?
    - Keep 'selection graph' with each node having only one goal, and pointing to the tactic_state it's derived from 
    - Nodes in selection graph only contain the subgoal itself and the context/assumptions. Choose from this, then from that create a tactic (with e.g. rotate to work off the original tactic_state to pass to LeanDojo)
    - When tactic_state returns, only add new subgoals as children to the new graph
    - Once goal proven in selection graph, will need some logic to reconstruct proof, as can't just concat edges
  - Steps:
    - Create selection tree with a node as a single goal
    - Apply tactic, get back subgoals
    - Parse tactic state to separate subgoals, add new nodes
    - Select with algorithm
    - From selection, create tactic to rotate left or right depending on base tactic state (base it on the first state the subgoal appears, or the smallest tactic_state it appears in??)
    - Track proven subgoals as before, and if overall goal is proven need to construct script manually 
    - Main point: Everything as before, except rotate/process selected tactic for the 'raw' tactic_state on which the selection node is based
  - Why?
      - Having full tactic state with all goals limits the context of the LLM. Keeping a single goal 
    as context could increase number of premises and focus the agent on one problem at a time. 
        - More relevant premises from retrieval, selecting premises only for the goal at hand rather than factoring in subgoals
      - Training data has few examples of multiple goals as context (5k vs 30k actually being generated)
      - Allows for more refined proof search strategy based on subgoals (e.g. Fringe, HTPS)
        - Enable open source HTPS implementation and comparison of strategies
    - Why not?
      - Some overhead, with rotate_tac 
        - Can't necessarily keep a tactic_state with the chosen goal at the top, since we may not have seen one
      - Prevents multi step proofs (agent proving multiple goals at once)
      - Possibly useful to have other subgoals in context when choosing strategy for a given goal

## Synthetic data/ end to end training
- Seems like Retriever and TacGen are trained separately without shared weights?

- Need information for three modules: Goal selection, retriever, tactic gen
  - Goal:
    - Track visit count per (sub)goal, if proven label 1 for provable, (or label number for num_steps to prove?)
    - If (sub)goal proven, track premises and tactic with label 1, otherwise can keep for hard negatives
  - Retriever
    - Track the state and the retrieved premises from the retriever model
      - Hard to tell true positive. 
      - HOList approach of retry the proof, with combination of premises at a time and see if proof successful
        - Doesn't really apply here since premises may be useful context, even if it's not used in the tactic 
      - For now, simple approach of if a premise was selected, and the goal was proven, then it's a positive label
        - Negative sampling as before (just use same file/allowed premises as main goal for all states in a proof attempt)
        - > Better way for negatives? Common problem across benchmarks, forms a *positive unlabelled* learning problem. 
          >>  See e.g. positive unlabelled contrastive learning, or Collective Relevance Labeling for Passage Retrieval
        - > For contrastive learning, need a transformation which preserves labels. Could take the transformation to 
           just be a child state with only one goal (i.e. so the goal is equivalent to the original, therefore class preserving.
           Scale e.g. by the number of proof steps away from original goal, so it's relatively close in difficulty)
  - Tactic
    - Easiest, just record state before (with/without premises), tactic for only proven (sub)goals, and take seq2seq loss

- Could label based on RLHF
  - For unproven goal, reward is e.g. -1 * number of visits
  - For proven goal: 
    - Want to have very high reward for any proof, but scaled down somewhat for longer proofs
    - Need to be careful, want to make sure that inherently harder proofs (which are longer) are rewarded, but also punish long proofs when a shorter proof is possible
    - Possible approach: Make ranking scheme for a given node that is proved:
      - Keep list of all goals proven, with tactic state and context. 
      - For a given goal, track all found proofs and re-rank according to best (shortest) proof 
      - Weight goals overall by their difficulty, e.g. wrt length of best found proof or total visits 
      - Scale reward by the difficulty of proof, and ranking within other proofs of the same goal
 
## Retriever architecture
- Could also compare GNN/SAT with LLM for retriever step (need graph representation first)
  - How to deal with tokenisation? Byte encoder doesn't require fixed tokenisation, would generalise better to new problems
    - Would need to parse graph first, then use byte representation of token?
  - 
- Generate tactics using different sets of premises. 
  - E.g 8 with top 8 premises, 8 with premises 8-16 etc.
  - Some other heuristic..


## New Proof search algo
  - Can keep separate threads for the tactic/goal selection and the environment running them
   
- Current search runs model on one state at a time, inefficient 
- Also runs all tactics from selection in environment
- Ideally LeanDojo would be parallelisable for a given goal (it isn't), but can still parallelise over different proofs 
 
- Idea:
  - Run Model on a given goal once to get all tactics
  - Rather than run all tactics sequentially, keep a ranked queue of (goal, tactic) tuples
    - Also track visit count of each goal
    - Can take e.g. product of goal and tactic, or weighted/learned combination of scores, with visit count factor
    - Intention is to be more sample efficient, as running e.g. 64 tactics per goal before trying a new goal is inefficient.
      - Want a model to be able to weight breadth vs depth, and consider other goals needed to be proven
    - HTPS, UpDown etc. applicable here 
  - Separate threads for model and environment interaction
    - Batch multiple goals for model at a time (memory dependent)
    - Need to check before running in environment if goal status is now proven

### Other ideas
- Subgoal search for complex reasoning tasks (generate possible future subgoals to help guide search). Used with BestFS, BFS and MCTS
- Curriculum learning (proof size bucket, goal is scored by distance to proof. Note this is same setup as LeanDojo 
without separating subgoals). Possibly better to extend unproven goals from being infinite length to being e.g. negative visit_count, proven goals being (max_size - proof_len)
which will maximally separate shorter proofs from unproven 
- HTPS (learns P(Provable | g, critic), with ground truth given by proven goals and policies estimate)
  - Having ground truth as policy estimate would reduce effectiveness for other models
  - Something like buckets would be more ideal.
  
## Repeated tactics
Often, tactics which are semantically similar/identical will be repeated by the model.
For example, intros x y vs intros a b. This results in very similar states which wastes search resources.

Possible solution: Condition the tactic generator on currently attempted tactics,
with the objective to try a different exploratory strategy. Or autoregressive tactic generation



## Results
- Original ReProver (no retrieval): pass@1 = 0.583, 1148 proved, 821 failed, 31 non-theorems discarded


