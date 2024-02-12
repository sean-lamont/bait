# Environments
HOL4 - Implemented, not tested
HOList - Implemented, tested to run, not tested fully
LeanDojo - Implemented, tested across several scenarios

# Models
## Generative
seq2seq - Implemented, tested live + training
DPO - Implemented, tested live + training
ILQL - Implemented, tested training

Retriever - Implemented, tested training

HOList - Implemented, not tested 
TacticZero - Not implemented

## Search
BestFS - Implemented, visualised, tested
UpDown - Implemented, visualised, tested partly
HTPS - Implemented, visualised, tested partly
BFS - Implemented, visualised, not tested
Fringe - Implemented, not visualised, not tested

# Features
- Distributed evaluation
  - configurable distribution across CPU and GPU resources for tactic, search and environment
- Constant format search trace
  - Independent of environment 
