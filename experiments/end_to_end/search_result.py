from typing import Dict, Any

from experiments.end_to_end.proof_node import *


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Any  # Any type to allow for different environments
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)

    # Proof search statistics.
    total_time: float
    tac_time: float
    search_time: float
    env_time: float
    num_expansions: int
    num_nodes: int

    # Ordered trace of edges, includes selected goal, outcome, tactic prob and goal probs
    trace: Any = field(repr=False)

    # Any additional data from the proof, to allow flexibility for different search / tactic generation setups
    data: Any = field(repr=False, default=None)
