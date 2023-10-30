from typing import Dict, Any

from refactor.proof_node import *


# todo add system (e.g. Lean , HOL4, etc.)
@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""
    theorem: str
    status: Status
    proof: Optional[List[str]]
    tree: Node
    nodes: Dict = field(repr=False)

    # Some statistics during proof search.
    total_time: float
    tac_time: float
    search_time: float
    env_time: float
    num_expansions: int
    num_nodes: int

    # Search trace to reconstruct state+selected goals+probs
    search_trace: Any = field(repr=False)

    # Tac gen trace
    tac_trace: Any = field(repr=False)
