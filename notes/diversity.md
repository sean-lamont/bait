# Diversity paper notes


## Abstract
Automated theorem proving blurb...

Current search based methods in AI-ITP suffer from an exponential growth in the number of proof paths. With
the execution of tactics in the environment being expensive, this limits the depth of the search and
hence the complexity of the problems that can be solved efficiently. Despite this, many tactics are semantically
similar, or lead directly to an execution error, which results in a large number of redundant tactic applications and proof
paths.

We propose a method to effectively prune this search, using only synthetic data generated from
previous proof attempts. We first show that it is possible to generate semantically aware tactic representations,
which capture the likelihood of success, execution time, and effect on the environment. 
We then propose a filtering mechanism to generate a semantically diverse set of tactics
which are both effective and efficient, through the use of Determinantal Point Processes.
Our approach is designed to be general and augment any tactic generation method, such as LLMs or RL agents.
We demonstrate the effectiveness of our approach by improving upon the base ReProver LLM on the miniF2F and LeanDojo benchmarks.


[//]: # (Current state of the art approaches leverage LLMs for the tactic generation, which are then used to search )
[//]: # (for a final proof.)
[//]: # (These LLMs are effective at generating a very large list of possible candidates, however it is intractable to try them all.)
[//]: # (Both due to the environment, and the exponential growth in proof paths.)

## Intro

Many of the generated tactics are semantically similar, varying with only, for example, variable
renaming.
See fig. x for a sample search tree from ReProver, where several semantically similar paths are explored, wasting
valuable resources.
Simple lexical similarity scores fail to capture the rich semantics which can be captured within a tactic. For example,
a complex expression is negated with only a single character.
It is therefore desirable to have a method to filter tactics by their diversity with respect to their semantics.
This can be captured by the effect of the tactic on the environment. Through generating a tactic vector which
can encode this, we can take the cosine similarity of such vectors to be a measure of their semantic similarity.

Of course, we do not with to have a diverse set of tactics which are ineffective. We wish to have a tradeoff between
the quality of the tactics and their semantic diversity. To achieve this, we leverage Determinantal Point Processes.

[//]: # (As noted by .., there are two primary categories)
[//]: # ( of approaches. The first generate tactics, and use the resulting states to form a search tree. The second tries)
[//]: # ( to generate the whole proof in a single attempt, with subsequent modifications based on environment feedback.)
[//]: # ( Despite the success of the second approach, there are advantages to maintaining a search tree.. &#40;smaller prompt/context, different candidate paths,...)


## Tactic Representation
### Tactic Embeddings
### Experimental setup
### Results
Single vs Combined vs All tokens for combined error/outcome/time task
#### Metrics
#### AutoRater

## Tactic Filtering
### Determinantal Point Processes
### Experimental setup
### Results
### Ablation

## Discussion
### Limitations
Environment timeouts depends on CPU

## Conclusion


## Figures
### Predictions from tactic embedding task
### Additional proofs discovered vs ReProver
### Search tree from ReProver with highlighted semantically similar paths
### Cosine similarity of tactics autoencoder vs tactic embeddings






