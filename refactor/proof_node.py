'''
Abstract proof state containing all relevant information for search.
Should include:
- goal/state (either single goal or a full state)
- status (proven, failed, open)
- is_explored/is_explored
- distance to proof
- visit count
- out_edges
- in_edges
- context
- ancestors
- optional provable/up score
- children (for computing total visit count)
- depth
- methods for extracting proof, updating context, status, ancestors,
'''

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Optional, List, Union, Set


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

    @property
    @abstractmethod
    def up_score(self) -> float:
        "Score representing the best probability of proving the given goal, based on child paths"
        raise NotImplementedError

    @property
    @abstractmethod
    def provable_score(self) -> float:
        "Probability of proving the goal in isolation, not considering children. Used to initialise up_score, and to compute the context-weighted score per goal"
        raise NotImplementedError


@dataclass(frozen=True)
class GoalFinished:
    message: Optional[str] = field(default=None, compare=False)


# Error in proof tree
@dataclass(frozen=True)
class TreeError:
    message: Optional[str] = field(default=None, compare=False)


@dataclass(frozen=True)
class EnvironmentError:
    message: Optional[str] = field(default=None, compare=False)


@dataclass
class ProofFinishedNode(Node):
    inner: GoalFinished
    status = Status.PROVED
    distance_to_proof = 0
    visit_count = 0
    is_terminal = True
    up_score = 0
    provable_score = 0


@dataclass
class ErrorNode(Node):
    inner: Union[EnvironmentError, TreeError]
    # inner: Union[LeanError, TimeoutError, ProofGivenUp, TreeError]
    status = Status.FAILED
    distance_to_proof = math.inf
    visit_count = 0
    is_terminal = True
    up_score = -math.inf
    provable_score = -math.inf


@dataclass(unsafe_hash=True)
class InternalNode(Node):
    """
    An internal node in the search tree, representing a nonterminal state.
    """

    # Unique goal, can either be isolated goal or complete state
    goal: str = field(compare=True)

    # The sum of action logprobs along edges from the root to this node
    cumulative_logprob: float = field(compare=False, repr=False)

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
    # _status is recomputed on an as-needed basis by children, since proving a child may prove this node.
    _status: Status = field(default=Status.OPEN, init=False, compare=False, repr=True)

    is_terminal = False

    # Number of steps separating this node from the end of a proof along the
    # optimal path. If unproved, infinity. Updated as needed by children.
    _distance_to_proof: float = field(
        default=math.inf, init=False, compare=False, repr=False
    )

    # Number of tactic applications from this node
    visit_count: int = field(default=0, compare=False, repr=False)

    # Scores based on the intrinsic probability of proving a goal, and the best available path from children
    provable_score = -math.inf
    up_score = -math.inf

    # Set flag to ignore this node in search
    is_explored: bool = field(default=False, compare=False, repr=False)

    # The depth of the node with respect to the initial goal. Currently only tracks the depth from the first parent.
    depth: int = field(default=0, compare=False, repr=False)

    max_expansions: int = field(default=64, compare=False, repr=False)

    @property
    def out_edges(self):
        return self._out_edges

    # This setter adds edges to the node, and updates the status
    # todo move context, ancestor updating to here?
    @out_edges.setter
    def out_edges(self, out_edges):
        self._out_edges = out_edges
        self._recompute_status()
        self._recompute_distance_to_proof()

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
                        sib_ctx = {g.goal for g in edge.dst if g != node}
                        node.add_context([ctx | sib_ctx for ctx in new_contexts])

    def add_ancestors(self, new_ancestors: Set[str]):
        if new_ancestors.issubset(self.ancestors):
            return

        self.ancestors = self.ancestors | new_ancestors
        if self.out_edges:
            for edge in self.out_edges:
                for node in edge.dst:
                    if isinstance(node, InternalNode):
                        node.add_ancestors(new_ancestors)

    # called when a node is proven, so the node can be ignored from search and propagated to all descendants
    def _set_explored(self):
        self.is_explored = True

        if self.out_edges:
            for edge in self.out_edges:
                for d in edge.dst:
                    if isinstance(d, InternalNode):
                        d._set_explored()

    def _recompute_status(self):
        """
        Recursively update the status of the current node and its ancestors.
        """
        # assert self.is_explored and self.out_edges is not None
        assert self.out_edges is not None

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

            self._set_explored()

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

        proving_edge = min(
            self.out_edges,
            key=Edge.distance_to_proof,
        )

        if all(child.is_terminal for child in proving_edge.dst):
            # Base case: this edge is all that's required to finish the proof
            assert all(isinstance(child, ProofFinishedNode) for child in proving_edge.dst)
            assert all(child.status == Status.PROVED for child in proving_edge.dst)
            return [proving_edge]
        else:
            # Recursive case: prove the child, then add this edge
            assert all(isinstance(child, InternalNode) for child in proving_edge.dst)
            assert all(child.status == Status.PROVED for child in proving_edge.dst)

            proof = []
            # add child proofs, expected to be in the correct order
            # for child in sorted(proving_edge.dst, key=lambda x: x.goal_num):
            # (should be sorted already) todo check
            for child in proving_edge.dst:
                assert isinstance(child, InternalNode)
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
        #     for edge in self.out_edges:  # type: is_explored
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

    def visit_count(self) -> int:
        return sum([d.visit_count for d in self.dst])
