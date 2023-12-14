## Diffusion/Subgoal paper

https://arxiv.org/pdf/2305.16366.pdf


- High level, uses LLM (ChatGPT) to construct proofs using informal subgoals, which are filled in formally and checked 
with the ITP
  - Start off with Initial goal, then model predicts candidate subgoals and formally fills them in
  - ITP verifies progress of each subgoal, if it's wrong, sent back as a prompt to change the subgoal
  - Repeats until each of the subgoals and their formal implementation is correct in the ITP
- Large component of success is adding in-context examples to the LLM prompt 
  - Shows the LLM relevant successful examples with the subgoal-formal format above
  - Choosing these is done with a diffusion model
  
- In-context examples 
  - ReProver just makes the in-context examples the nearest cosine premises
  - Can extend this to include some NLP summary/subgoal?
    - Need training data for this, could look at e.g. autoformalization datasets?