# Random notes for large model experiments
## Generation
- Beam search / multiple sampled candidates are harder to fit in memory
- Can avoid by resampling a smaller number and iterating until k unique samples
- Very slow! ~2m for one goal with 16 tactics vs ~2s for 64 tactics with small model
- Other option is to have model generate several options from a single sampling
  - Still slow (less so), but then have longer context length
  - Also harder to finetune on 
- Beam search not ideal for diverse sample generation
  - Todo: beam groups from HF?
- Can see this through various examples with e.g. variable renaming
- Want some way to generate diverse samples with different 'proof paths'
  - NLP DSP does this to some degree with NLP plan, then formal fill-in

## Quantization / LoRA
- Quantization (4 bit) has little impact in training (from loss curves)
- Haven't been able to avoid (memory)

## Ideas 
- Search
  - Better Goal model
  - custom prompt to guide search?
- Dataset Generation
  - Error solving tactics
- Filtering
  - Removing tactics which are semantically similar from a large list
- Learning from negatives
  - Prompt based (Uni melb paper)
- Environment dynamics
    - E.g. predict what state will result from tactic x
     

- Time difference in generation may make it worthwhile to have many calls to a small LM to generate, 
followed by filtering from a larger model
  - Either local or remote Large model
  - Finetune? 
    - Can be done with environment dynamics? 