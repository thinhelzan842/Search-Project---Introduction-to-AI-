from abc import ABC, abstractmethod

class ProblemBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, solution) -> float:
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
    def get_neighbors(self) -> Any: #this is for search in discrete problems
        pass

    @abstractmethod
    def is_goal(self, solution) -> bool:
        pass

    @abstractmethod
    def get_bounds(self) -> List[Tuple[float, float]]:
        pass

class AlgorithmBase(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self, problem):
        pass