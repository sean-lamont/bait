# Diversity paper notes

## Abstract

Automated theorem proving blurb...

Current search based methods in AI-ITP suffer from an exponential growth in the number of proof paths. With
the execution of tactics in the environment being expensive, this limits the search depth and
hence the complexity of the problems that can be solved efficiently. Despite this, many tactics are semantically
similar, or lead directly to an execution error, which results in a large number of redundant tactic applications and
proof
paths.

We propose a method to effectively prune this search, using only synthetic data generated from
previous proof attempts. We first show that it is possible to generate semantically aware tactic representations,
which capture the likelihood of success, execution time, and effect on the environment.
We then propose a filtering mechanism to generate a semantically diverse set of tactics
which are both effective and efficient, through the use of Determinantal Point Processes.
Our approach is designed to be general and augment any tactic generation method, such as LLMs or RL agents.
We demonstrate the effectiveness of our approach by improving upon the base ReProver LLM on the miniF2F and LeanDojo
benchmarks.


[//]: # (Current state of the art approaches leverage LLMs for the tactic generation, which are then used to search )

[//]: # (for a final proof.)

[//]: # (These LLMs are effective at generating a very large list of possible candidates, however it is intractable to try them all.)

[//]: # (Both due to the environment, and the exponential growth in proof paths.)

## Intro
- Search tree from ReProver with highlighted semantically similar paths and error paths
  - Have e.g. intro x intro y intro z, with some errors with red lines

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


Learning from failed proof attempts is a difficult task,
and to date there has been little work addressing this.
Methods such as HTPS, Co-training, curriculum learning all learn from
successful proof attempts, however the failed attempts are 
leveraged only to score a goal. The problem with this is
that most goals are failures, so there is little signal to enable 
efficently learning a goal model. To illustrate this, we 
found that x% of goals were unproven. Furthermore, an unproven 
node is not necessarily a failure, as it may be 
that further exploration would have resulted in a proof.

[//]: # (As noted by .., there are two primary categories)

[//]: # ( of approaches. The first generate tactics, and use the resulting states to form a search tree. The second tries)

[//]: # ( to generate the whole proof in a single attempt, with subsequent modifications based on environment feedback.)

[//]: # ( Despite the success of the second approach, there are advantages to maintaining a search tree.. &#40;smaller prompt/context, different candidate paths,...)

### Related Work
Co-training looks at auxiliary tasks, however it doesn't use synthetic data,
with nothing to augment the search/diversity. 

Emily et al. ? diversity in thereom proving paper, looks at an ensemble of models.
Models are separately trained, which is infeasible for large LMs, and no modification of search.

HTPS / ACL paper using MCTS like approach, only other area which looks at search directly.
Although it includes exploration, requires learning a goal model which is difficult. Also not open source
And nothing to do with diversity.

To our knowledge, no work on learning environment transitions directly. (check, e.g. model based theorem proving)


## Problem setup

The AI-ITP problem can be formulated as follows.

- Given the initial goal $g$, we wish to find a sequence of tactics
  $t_1, t_2, \ldots, t_n$ which when applied to the environment result in a complete proof of $g$.
- Although approaches exist which attempt to generate the whole proof in a single attempt \cite{todo},
  we restrict ourselves to the case where a single tactic is generated at a time.
- We assume a base model $\pi_{\theta}$ which generates a list of tactics $T$ for a given goal $g$.
- We further assume we have a filtering model $\pi_{\phi}$ which takes the list of tactics $T$ and returns a subset $t$.
- ...
- Following a proof attempt, we have a trace of the (goal, tactic, outcome) tuples, which we use to train our models.
- The outcome of a tactic results in a three-tuple: (status, time, result)

## Tactic Representation

As discussed in the previous section, we have a dataset of (goal, tactic, outcome) tuples,
where the outcome is a three-tuple: (status, time, result).

Previous work has focused on learning only with the status component, for example by
using a reward, seq2seq training over proven status nodes, and by learning a goal scoring function
which assigns a score to a goal based on the predicted status.

This discards a large amount of information about how a tactic affects the environment.
In this section, we investigate whether it is possible to learn these transitions, and
further investigate if this can be encoded in a single tactic representation vector.

Learning the transition function is a difficult task, as it requires understanding of both the
environment, goal and tactic.

For our primary tactic encoding model, we start with an Encoder-Decoder Transformer.
The tactic is concatenated with the goal, and the embeddings from the Encoder are computed for all tokens.
We then generate a single tactic embedding by mean pooling over the tactic tokens.
This single tactic embedding is then used for the three outcome prediction tasks:

- Status prediction (classification)
- Time prediction (regression)
- Result prediction (autoregressive)

The status and time prediction tasks share a single-layer MLP, which takes this tactic embedding as input.
The MLP outputs two final values for the respective tasks, with the status prediction output being passed through a
final sigmoid activation.
We use a binary cross-entropy loss for the status prediction task, and a mean squared error loss for the time prediction
task.

For the result prediction task, we concatenate the tactic embedding with the goal token embeddings, and pass this
through the entire Encoder-Decoder Transformer. The loss is calculated using a cross-entropy loss between the predicted
and actual result tokens.

We compare this with a model which encodes the tactic separately from the goal.
We hypothesise that allowing the tactic tokens to attend to the goal tokens will allow the model to better understand
the semantics of the tactic.

We finally compare with a model which uses all tactic tokens, without reducing to a single embedding.

### Tactic Embeddings
- Tactic embedding architecture figure

### Experimental setup

- 2 x A6000 GPUs with 48GB memory each, etc..
- AdamW optimizer, learning rate 1e-5, batch size 4, etc..

### Results

Single vs Combined vs All tokens for combined error/outcome/time task
Also no tactic to show baseline for comparison

Show best val score for BLEU, ROGUE, error acc, time MSE, top-K
(take best model from each run, test on the whole val set)

#### Metrics

- Report BLEU, ROGUE, error accuracy, time MSE, top-K

#### AutoRater

## Tactic Filtering

### Determinantal Point Processes

Determinantal Point Processes (DPPs) are a class of probabilistic models which sample
subsets $A$ from a ground set $\mathcal{X}$.
They provide an elegant way of sampling diverse but high quality subsets, by constructing a
likelihood matrix $L$ which captures the similarity between elements in $\mathcal{X}$.
The probability of sampling a subset $A$ is then proportional to the determinant of the submatrix of $L$ indexed by $A$.
Geometrically, it is the volume of the parallelepiped spanned by the rows of $L$ indexed by $A$.
The larger the diversity of the elements in $A$, the larger the volume, and hence the higher the probability of sampling
$A$.

- Diversity search model architecture diagram
 
### Experimental setup

### Results
- Pass@1 (over averages?)
- Union over approaches
- End-to-end / cumulative
- Average number of errors 
- Average time per tactic
- Distribution of number of tactics selected per goal

### Ablation

## Discussion

### Limitations

Environment timeouts depends on CPU

### Future work

- Model-based approach (is uncertain to work even with a good transition function, as
  any errors will compound over the number of rollouts)
- Co-training style approach using only synthetic data (i.e. not separating models for the main and auxiliary tasks)

## Conclusion

## Figures

- Predictions from tactic embedding task

- Additional proofs discovered vs ReProver

- Search tree from ReProver with highlighted semantically similar paths

- Cosine similarity of tactics autoencoder vs tactic embeddings

- Tactic embedding architecture

- Diversity search model architecture






