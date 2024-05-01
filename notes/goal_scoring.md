# Goal scoring

- Model for scoring goals, used in HTPS, Fringe (TacticZero) and BestFS
 
- Options to train the model:
  - Proof length weighted sum (Polu et al.) (trained from proof attempts)
  - Provable/Unprovable (HTPS), trained with seq2seq over tokens (with soft labels from proof attempts)
  - Score function (Fringe), trained with policy gradient
  - Take into account visit count, scale from 0 to 1 (high VC, no proof -> 0, low VC, proof -> 1, low VC, no proof -> 0.5)

- Problems:
  - Imbalanced data (most goals aren't proved)
    - Positive unlabelled learning ?
  - Hard to evaluate (need to run the prover)
  - Sweep over hyperparameters (exploration constant, visit count cutoffs)

- To compare approaches, the same underlying goal model would be helpful
  - E.g. trained on the same provable/unprovable, used for HTPS, Fringe
  - Need to decide on the best approach for training the model
    - Compare provable/unprovable, proof length, vc scaling, run 1-2 experiments with each 
  - Should we be able to look at a proof trace independent of the approach, and train a model from this?
    - HTPS doesn't, as it uses the model to determine the targets
  

- Paper ideas (5 months to ICLR):
  - If we can improve on HTPS/BestFS use that
  - Otherwise paper would focus on thorough comparison of Fringe, HTPS, BestFS, BFS
  - Would compare training approaches (proof length, provable/unprovable, vc scaling)
  - Further would explore hyperparameter influence (exploration constant, num nodes expanded, allowing dynamic expansion)
  - Investigate meta-controller vs random sampling
  - Investigating goal model a largely unexplored area
    - Show that current approaches all use the idea of a goal model, which is somewhat independent of the search algorithm
    - Hasn't been thoroughly compared in the literature
    - Important to isolate this from the search algorithm to understand the impact of the goal model and the best way to train it

## Pairing proven with similar goals
Idea: when a node is proven, we look at goals with the same parent, and pair them with the proven goal.
Then we have pairs of (positive, negative) examples. If we want multiple negatives for each positive, 
we can use e.g AUROC loss from HOList premise selection, which pairs a single positive with multiple negatives.

- Can run over human proofs with good ground truth for the proven goals
- Can run over successful generated proofs
- Won't work with failed proofs
  - Could use unexplored nodes as positives, and highly visited nodes as negatives, but would be noisy
- Can generate pairs with error nodes
  - If a node is a failure, take a sibling which aren't failures as positive

Summary:
- Generate pairs with:
  - Proven goals (as positive), and unproven goals with the same parent
  - Failed node as negative, and open siblings as positive
  - Highly visited nodes as negative, and unvisited nodes as positive
  - Can scale the loss, with positives being weighted highly, negatives weighted middling, and unvisited nodes weighted lowly
    - Or both positive and negative weighted highly, with unvisited nodes weighted lowly, since these are the most uncertain
  - Keep provable/unprovable token. 
  - One forward pass will have a single pair, 
    the loss is the difference between logits for the provable token of the positive and negative example (auroc)
    - log softmax over logits first? 
  - For validation / inference, take score as the softmax over the logits for the provable token

[//]: # (Could also use something like this rather than seq2seq for training the tactic generation model?)

### How to compare?
- Keep constant validation set of failures/proofs and see which performs best (we know these to be ground truth)
