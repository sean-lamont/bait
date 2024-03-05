from typing import Dict, Any

from experiments.end_to_end.proof_node import *


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    # Any type to allow for different environments
    theorem: Any
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)
    # Environment used to prove the theorem

    # Some statistics during proof search.
    total_time: float
    tac_time: float
    search_time: float
    env_time: float
    num_expansions: int
    num_nodes: int

    # ordered trace of edges, includes selected goal, outcome, tactic prob and goal probs
    trace: Any = field(repr=False)

    # any additional data from the proof
    data: Any = field(repr=False, default=None)
