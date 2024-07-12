
## Diversity paper notes

 Automated theorem proving blurb...

Current state of the art approaches leverage LLMs for the tactic generation, which are then used to search 
for a final proof.

[//]: # (These LLMs are effective at generating a very large list of possible candidates, however it is intractable to try them all.)
[//]: # (Both due to the environment, and the exponential growth in proof paths.)

As this grows exponentially, there are large performance gains to be obtained through 
effective pruning. Many of the generated tactics are semantically similar, varying with only, for example, variable renaming.
See fig. x for a sample search tree from ReProver, where several semantically similar paths are explored, wasting valuable resources.
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

