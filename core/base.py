from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Iterable

T = TypeVar('T') 

class ProblemBase(ABC, Generic[T]):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def evaluate(self, solution: T) -> float: #min-optimization
        pass

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


class ContinuousProblem(ProblemBase[List[float]], HasBounds, GeneratesRandom[List[float]]):
    def get_optimal_value(self) -> float:
        return 0.0

class DiscreteSearchProblem(ProblemBase[T], HasNeighbors[T], HasGoal[T]):
    @abstractmethod
    def get_initial_state(self) -> T:
        pass

    def get_optimal_value(self) -> float:
        if getattr(self, '_optimal_value', None) is not None:
            return self._optimal_value

        from algorithms import UCS

        print(f"      [System] Calculating true optimum for {self.name()} using UCS...")
        ucs_solver = UCS()
        best_val = float('inf')

        try:
            for state in ucs_solver.run(self):
                best_val = state.get('best_score', best_val)
            self._optimal_value = best_val
        except Exception as e:
            print(f"      [Warning] UCS failed to find optimal value: {e}")
            self._optimal_value = 0.0  # fallback

        return self._optimal_value

class AlgorithmBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, problem):
        pass