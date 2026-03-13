from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Iterable

# T represents the State/Solution type (e.g., List[float], int, Tuple[int, int])
T = TypeVar('T') 

class ProblemBase(ABC, Generic[T]):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def evaluate(self, solution: T) -> float: 
        """Always scaled for min-optimization."""
        pass

# --- Mixins/Traits (Capabilities) ---

class HasBounds(ABC):
    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        pass

class GeneratesRandom(ABC, Generic[T]):
    @abstractmethod
    def random_solution_generate(self) -> T:
        pass

class HasNeighbors(ABC, Generic[T]):
    @abstractmethod
    def get_neighbors(self, current_state: T) -> Iterable[T]:
        pass

class HasGoal(ABC, Generic[T]):
    @abstractmethod
    def is_goal(self, solution: T) -> bool:
        pass

class HasHeuristic(ABC, Generic[T]): #admissible heuristic for A* and similar algorithms
    @abstractmethod
    def get_heuristic(self, solution: T) -> float:
        pass

# --- Concrete Base Types ---

class ContinuousProblem(ProblemBase[List[float]], HasBounds, GeneratesRandom[List[float]]):
    """Algorithms like standard ABC or Continuous GA will expect this interface."""
    pass

class DiscreteSearchProblem(ProblemBase[T], HasNeighbors[T], HasGoal[T]):
    """Algorithms like A*, BFS, DFS will expect this interface."""
    @abstractmethod
    def get_initial_state(self) -> T:
        pass

class AlgorithmBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, problem):
        pass