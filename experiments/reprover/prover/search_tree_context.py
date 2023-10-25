"""Definitions of the search tree used by the prover.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Optional, List, Iterable, Union, Tuple, Set

from lean_dojo import (
    TacticState,
    LeanError,
    # TacticError,
    TimeoutError,
    ProofGivenUp,
)
from loguru import logger


class Status(Enum):
    """Status of a node or a proof search."""

    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.


class Node(ABC):
    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def distance_to_proof(self) -> int:
        "The smallest number of steps to a proof."
        raise NotImplementedError

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def visit_count(self) -> int:
        "The number of times a tactic has been applied to this node or a descendant"
        raise NotImplementedError


@dataclass(frozen=True)
class GoalFinished:
    message: Optional[str] = field(default=None, compare=False)


# Error in proof tree
@dataclass(frozen=True)
class TreeError:
    message: Optional[str] = field(default=None, compare=False)


@dataclass
class ProofFinishedNode(Node):
    inner: GoalFinished
    status = Status.PROVED
    distance_to_proof = 0
    visit_count = 0
    is_terminal = True


@dataclass
class ErrorNode(Node):
    # inner: Union[TacticError, TimeoutError, ProofGivenUp, TreeError]
    inner: Union[LeanError, TimeoutError, ProofGivenUp, TreeError]
    status = Status.FAILED
    distance_to_proof = math.inf
    visit_count = 0
    is_terminal = True


@total_ordering
@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.

    Nodes are sorted by _inverse_ priority, for compatibility with the `heapq` library.
    That is, node_a < node_b is true if node_a has _higher_ priority than node_b.
    """

    # The high level goal to be proven. This implementation restricts nodes to have only one goal.
    # Only field used for hashing.
    goal: str = field(compare=True)

    # The first state where the goal was created. Used as a 'surrogate', where the tactic selected
    # for 'goal' is applied to 'state', with a rotation defined by 'goal_num' to ensure the correct goal is selected
    state: TacticState = field(compare=False, repr=False)

    # The sum of action logprobs along edges from the root to this node
    cumulative_logprob: float = field(compare=False, repr=False)

    # Tracks the number of the goal in the surrogate tactic state,
    # so the correct rotation can be applied in tactic application
    goal_num: int = field(compare=False)

    # Tracks the top level goal of siblings and ancestors required for a full proof.
    # Can include multiple possible paths, each of which will satisfy the requirements
    context: List[Set[str]] = field(compare=False, default_factory=list, repr=False)

    # Tracks all ancestor nodes which lead to this node, used to detect and prevent cycles
    ancestors: Set[str] = field(compare=False, default_factory=set, repr=False)

    # All edges known to lead to this node.
    # May change at any time as other nodes are explored.
    in_edges: List["Edge"] = field(
        default_factory=list, init=False, compare=False, repr=False
    )

    # All edges out of this node that we've considered, or None for unexplored nodes.
    # When a node is explored, this list is populated, and can be added to if backtracked
    _out_edges: Optional[List["Edge"]] = field(
        default=None, init=False, compare=False, repr=False
    )

    # A node is proved if any child is proved, and failed if it corresponds to an error node
    # A node that is proved or failed cannot change status
    # _status is recomputed on an as-needed basis by children,
    # since proving a child may prove this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    visit_count = 0

    # Maintain a list of unique child nodes, used to calculate the total visit count of a branch
    children: Set[str] = field(compare=False, default_factory=set, repr=False)

    @property
    def out_edges(self):
        return self._out_edges

    # This setter implements exploring this node
    @out_edges.setter
    def out_edges(self, out_edges: Iterable["Edge"]) -> Optional[List["Edge"]]:
        if self.is_explored:
            raise RuntimeError("Node is already explored.")

        self._out_edges = list(out_edges)
        self._recompute_status()
        self._recompute_distance_to_proof()

    # A node is considered explored if we've evaluated the actor in the node to generate
    # a list of candidate children.
    @property
    def is_explored(self) -> bool:
        return self.out_edges is not None

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, s):
        self._status = s

    def add_context(self, contexts: List[Set[str]]):
        new_contexts = []
        for context in contexts:
            if context not in self.context:
                new_contexts.append(context)

        if not new_contexts:
            return

        self.context.extend(new_contexts)

        if self.out_edges:
            for edge in self.out_edges:
                for node in edge.dst:
                    if isinstance(node, InternalNode):
                        node.add_context(new_contexts)

    def add_ancestors(self, new_ancestors: Set[str]):
        if new_ancestors.issubset(self.ancestors):
            return

        self.ancestors = self.ancestors | new_ancestors
        if self.out_edges:
            for edge in self.out_edges:
                for node in edge.dst:
                    if isinstance(node, InternalNode):
                        node.add_ancestors(new_ancestors)

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        assert self.is_explored and self.out_edges is not None

        # If this node is proved or failed, nothing can change that
        if self._status != Status.OPEN:
            return

        # If all subgoals from a child are proven, this node is proved, and so are parents recursively
        for edge in self.out_edges:
            if all(child.status == Status.PROVED for child in edge.dst):
                self._status = Status.PROVED

        # If this node was proved or failed, parents may need to recompute.
        # This is guaranteed to terminate because only open nodes can change, and
        # there are a finite number of open nodes in the tree.
        if self._status != Status.OPEN:
            for edge in self.in_edges:
                edge.src._recompute_status()

    @property
    def priority(self) -> float:
        return self.cumulative_logprob

    def __lt__(self, other: "InternalNode") -> bool:
        return self.priority > other.priority

    @property
    def distance_to_proof(self) -> float:
        return self._distance_to_proof

    def _recompute_distance_to_proof(self):
        """
        Recursively update the distance_to_proof of the current node and its ancestors.
        """
        if self.out_edges:
            distance = min(edge.distance_to_proof() for edge in self.out_edges)
        else:
            distance = math.inf

        if distance < self._distance_to_proof:
            self._distance_to_proof = distance
            for edge in self.in_edges:
                edge.src._recompute_distance_to_proof()

    def extract_proof(self) -> Optional[List["Edge"]]:
        """

        Extract a proof of the current node as a sequence of edges.

        """

        if self.status != Status.PROVED:
            return None

        assert self.is_explored

        proving_edge = min(
            self.out_edges,
            key=Edge.distance_to_proof,
        )

        if all(child.is_terminal for child in proving_edge.dst):
            # if proving_edge.dst.is_terminal:
            # Base case: this edge is all that's required to finish the proof
            assert all(isinstance(child, ProofFinishedNode) for child in proving_edge.dst)
            assert all(child.status == Status.PROVED for child in proving_edge.dst)
            return [proving_edge]
        else:
            # Recursive case: prove the child, then add this edge
            assert all(isinstance(child, InternalNode) for child in proving_edge.dst)
            assert all(child.status == Status.PROVED for child in proving_edge.dst)

            proof = []
            # add child proofs in correct order
            for child in sorted(proving_edge.dst, key=lambda x: x.goal_num):
                child_proof = child.extract_proof()
                assert child_proof
                proof.extend(child_proof)

            return [proving_edge, *proof]

    #########
    # Debug #
    #########

    # todo
    def check_invariants(self):
        """
        Perform some sanity checks.
        """
        return
        # if not self.is_explored:
        #     assert self.status == Status.OPEN
        #     return  # Nothing more can be said about unexplored nodes
        # #
        # for edge in self.in_edges:
        #     assert self in edge.dst  # edge.dst is self
        # #
        # if self.out_edges == []:
        #     assert self.status == Status.FAILED
        # else:
        #     for edge in self.out_edges:  # type: ignore
        #         assert edge.src is self
        # #
        # if self.status == Status.PROVED:
        #     assert self.out_edges
        #     # assert all(edge.dst.status == Status.PROVED for edge in self.out_edges)
        #     assert any([all(child.status == Status.PROVED for child in edge.dst) for edge in self.out_edges])
        #     # assert all(edge.dst.status == Status.PROVED for edge in self.in_edges)
        #
        #     proof_by_steps = self.extract_proof()
        #     assert proof_by_steps is not None
        #     assert self.distance_to_proof == len(proof_by_steps)
        # #
        # # elif self.status == Status.FAILED:
        # #     assert self.out_edges is not None
        # #     assert all(edge.dst.status == Status.FAILED for edge in self.out_edges)
        # #     assert self.distance_to_proof == math.inf
        # #     assert self.extract_proof() == None
        # #
        # elif self.status == Status.OPEN:
        #     assert self.out_edges
        #     # assert not any(edge.dst.status == Status.PROVED for edge in self.out_edges)
        #     assert not any([all(child.status == Status.PROVED for child in edge.dst) for edge in self.out_edges])
        #     # assert not all(edge.dst.status == Status.FAILED for edge in self.out_edges)
        #     assert self.distance_to_proof == math.inf
        #     assert self.extract_proof() == None


@dataclass
class Edge:
    """An edge in the search tree, representing a tactic.
    This implementation keeps a set of child nodes,
    corresponding to separate goals which need to be proven to prove the source
    """

    tactic: str
    src: InternalNode = field(repr=False)
    dst: List[Node] = field(repr=False)
    logprob: float
    time: float

    def distance_to_proof(self) -> float:
        return 1 + sum([d.distance_to_proof for d in self.dst])
