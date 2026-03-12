from abc import ABC, abstractmethod
from typing import Any, List, Tuple

"""class ProblemBase(ABC):
    @abstractmethod
    def evaluate(self, solution) -> float: #scale so that it is always min-optimization
        pass

class Boundable(ABC):
    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        pass

class Searchable(ABC):
    @abstractmethod
    def get_neighbors(self, progress) -> Any:
        pass

    @abstractmethod
    def get_initial_state(self) -> Any:
        pass

class AlgorithmBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, problem):
        pass"""

class ProblemBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def evaluate(self, solution) -> float: #scale so that it is always min-optimization
        pass

    @abstractmethod
    def is_discrete(self) -> bool:
        pass

    @abstractmethod
    def is_min_optimization(self) -> bool:
        pass

    @abstractmethod
    def random_solution_generate(self) -> Any: #assist algorithms like genetic algorithm, etc
        pass

    @abstractmethod
    def get_initial_state(self) -> Any:
        pass

    @abstractmethod
    def get_neighbors(self, progress) -> Any: #this is for search in discrete problems
        pass

    @abstractmethod
    def is_goal(self, solution) -> bool:
        pass

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        pass

class ContinuousProblemBase(ProblemBase):
    def is_discrete(self) -> bool:
        return False

    def get_neighbors(self) -> Any:
        raise NotImplementedError("Continuous problems don't support returning neighbors")

    def get_initial_state(self) -> Any:
        return self.random_solution_generate()
    
class DiscreteProblemBase(ProblemBase):
    def is_discrete(self) -> bool:
        return True

    def get_bounds(self) -> Any:
        raise NotImplementedError("Discrete problems don't support returning bounds")

class AlgorithmBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, problem):
        pass