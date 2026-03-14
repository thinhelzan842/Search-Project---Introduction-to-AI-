import numpy as np
import random
from core import AlgorithmBase

class GeneticAlgorithm(AlgorithmBase):
    def __init__(self, size, gen, desire=None, mutate_prob=0.1, crossover_prob=0.9,
                 tournament_size=3, crossover_type='one_point', mutation_type='bit_flip', sigma=0.1, points=2):
        self.size, self.gen, self.desire = size, gen, desire
        self.mutate_prob, self.crossover_prob = mutate_prob, crossover_prob
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.sigma, self.points = sigma, points

    def name(self) -> str: return f"GA ({self.crossover_type} + {self.mutation_type})"

    # --- CROSSOVER STRATEGIES ---
    def _crossover_one_point(self, dad, mom):
        if len(dad) < 2: return list(mom)
        pt = random.randint(1, len(dad) - 1)
        return list(dad[:pt]) + list(mom[pt:])

    def _crossover_multi_point(self, dad, mom):
        if len(dad) < 2 or len(mom) < 2: return list(dad)
        max_pts = len(dad) - 1
        pts = sorted(random.sample(range(1, len(dad)), min(self.points, max_pts)))
        child, last_pt = [], 0
        for i, pt in enumerate(pts):
            child.extend(dad[last_pt:pt] if i % 2 == 0 else mom[last_pt:pt])
            last_pt = pt
        child.extend(dad[last_pt:] if len(pts) % 2 == 0 else mom[last_pt:])
        return child

    def _crossover_order(self, dad, mom):
        if len(dad) < 2 or len(mom) < 2: return list(dad)
        a, b = sorted(random.sample(range(len(dad)), 2))
        child = dad[a:b]
        for item in mom:
            if item not in child:
                child.append(item)
        return child

    # --- MUTATION STRATEGIES ---
    def _mutate_bit_flip(self, child, right_bound=1):
        return [right_bound - gene if random.random() < self.mutate_prob else gene for gene in child]

    def _mutate_swap(self, child):
        if len(child) < 2: return child
        if random.random() < self.mutate_prob:
            idx1, idx2 = random.sample(range(1, len(child)), 2)
            child[idx1], child[idx2] = child[idx2], child[idx1]
        return child

    def _mutate_gaussian(self, child):
        return [gene + random.gauss(0, self.sigma) if random.random() < self.mutate_prob else gene for gene in child]

    def select(self, pop, scores):
        participants = random.sample(list(zip(pop, scores)), min(len(pop), self.tournament_size))
        # Strictly minimizing
        participants.sort(key=lambda x: x[1])
        return participants[0][0]

    def run(self, problem):
        self.pop = [problem.random_solution_generate() for _ in range(self.size)]
        scr_list = [problem.evaluate(v) for v in self.pop]
        
        # Get index of strictly lowest score
        best_idx = np.argmin(scr_list)
        best_sol, best_scr = self.pop[best_idx], scr_list[best_idx]

        yield {
            'iteration': 0, 'current_solution': self.pop.copy(), 'current_score': scr_list.copy(),
            'best_solution': best_sol, 'best_score': best_scr
        }

        gen, diff = 0, float('inf')

        while (self.desire is None or best_scr > self.desire) and gen < self.gen and diff > 0.1:
            new_pop = [best_sol]  # Elitism

            while len(new_pop) < self.size:
                dad = self.select(self.pop, scr_list)
                mom = self.select(self.pop, scr_list)

                # CROSSOVER
                if random.random() < self.crossover_prob:
                    if self.crossover_type == 'order': child = self._crossover_order(dad, mom)
                    elif self.crossover_type == 'multi_point': child = self._crossover_multi_point(dad, mom)
                    else: child = self._crossover_one_point(dad, mom)
                else:
                    child = list(dad)

                # MUTATION
                if self.mutation_type == 'swap': child = self._mutate_swap(child)
                elif self.mutation_type == 'gaussian': child = self._mutate_gaussian(child)
                else: child = self._mutate_bit_flip(child)

                new_pop.append(child)

            self.pop = new_pop[:self.size]
            scr_list = [problem.evaluate(v) for v in self.pop]

            new_best_idx = np.argmin(scr_list)
            new_best_sol, new_best_scr = self.pop[new_best_idx], scr_list[new_best_idx]
            
            diff = abs(new_best_scr - best_scr)
            best_sol, best_scr = new_best_sol, new_best_scr
            gen += 1

            yield {
                'iteration': gen, 'current_solution': self.pop.copy(), 'current_score': scr_list.copy(),
                'best_solution': best_sol, 'best_score': best_scr
            }

        return best_sol, best_scr

"""import numpy as np
import random
from core import *


class GeneticAlgorithm(AlgorithmBase):
    def __init__(self, size, gen, desire,
                 mutate_prob=0.1,
                 crossover_prob=0.9,
                 tournament_size=3,
                 crossover_type='one_point',  # 'one_point', 'multi_point', 'order'
                 mutation_type='bit_flip',  # 'bit_flip', 'swap', 'gaussian'
                 sigma=0.1,  # For Gaussian
                 points=2  # For Multi-point
                 ):
        self.size = size
        self.gen = gen
        self.pop = []
        self.desire = desire
        self.mutate_prob = mutate_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.sigma = sigma
        self.points = points

    def name(self) -> str:
        return f"GA ({self.crossover_type} + {self.mutation_type})"

    # --- CROSSOVER STRATEGIES ---

    def _crossover_one_point(self, dad, mom):
        if len(dad) < 2: return list(mom)
        pt = random.randint(1, len(dad) - 1)
        return list(dad[:pt]) + list(mom[pt:])

    def _crossover_multi_point(self, dad, mom):
        if len(dad) < 2 or len(mom) < 2:
            return list(dad)
        max_pts = len(dad) - 1
        pts = sorted(random.sample(range(1, len(dad)), min(self.points, max_pts)))
        child, last_pt = [], 0
        for i, pt in enumerate(pts):
            child.extend(dad[last_pt:pt] if i % 2 == 0 else mom[last_pt:pt])
            last_pt = pt
        child.extend(dad[last_pt:] if len(pts) % 2 == 0 else mom[last_pt:])
        return child

    def _crossover_order(self, dad, mom):
        #Standard OX1 for permutations; modified for variable length.
        if len(dad) < 2 or len(mom) < 2: return list(dad)
        a, b = sorted(random.sample(range(len(dad)), 2))
        child = dad[a:b]
        for city in mom:
            if city not in child:
                child.append(city)
        return child

    # --- MUTATION STRATEGIES ---

    def _mutate_bit_flip(self, child, right_bound=1):
        return [right_bound - gene if random.random() < self.mutate_prob else gene for gene in child]

    def _mutate_swap(self, child):
        if len(child) < 2: return child
        if random.random() < self.mutate_prob:
            # We skip index 0 to avoid swapping the starting city of a TSP path
            idx1, idx2 = random.sample(range(1, len(child)), 2)
            child[idx1], child[idx2] = child[idx2], child[idx1]
        return child

    def _mutate_gaussian(self, child):
        return [gene + random.gauss(0, self.sigma) if random.random() < self.mutate_prob else gene for gene in child]

    # --- SELECTION ---

    def select(self, pop, scores, is_min):
        participants = random.sample(list(zip(pop, scores)), min(len(pop), self.tournament_size))
        participants.sort(key=lambda x: x[1], reverse=not is_min)
        return participants[0][0]

    # --- MAIN LOOP ---

    def run(self, problem):
        is_min = problem.is_min_optimization()
        self.pop = [problem.random_solution_generate() for _ in range(self.size)]

        def get_best_stat(p, s):
            combined = list(zip(p, s))
            return min(combined, key=lambda x: x[1]) if is_min else max(combined, key=lambda x: x[1])

        scr_list = [problem.evaluate(v) for v in self.pop]
        best_sol, best_scr = get_best_stat(self.pop, scr_list)

        yield {
            'generation':       0,
            'current_solution': self.pop.copy(),
            'current_score':    scr_list.copy(),
            'best_solution':    best_sol,
            'best_score':       best_scr
        }

        gen, diff = 0, float('inf')
        def target_met():
            if self.desire is None:
                return False
            return best_scr <= self.desire if is_min else best_scr >= self.desire

        while not target_met() and gen < self.gen and diff > 0.1:
            new_pop = [best_sol]  # Elitism

            while len(new_pop) < self.size:
                dad = self.select(self.pop, scr_list, is_min)
                mom = self.select(self.pop, scr_list, is_min)

                # CROSSOVER
                if random.random() < self.crossover_prob:
                    if self.crossover_type == 'order':
                        child = self._crossover_order(dad, mom)
                    elif self.crossover_type == 'multi_point':
                        child = self._crossover_multi_point(dad, mom)
                    else:
                        child = self._crossover_one_point(dad, mom)
                else:
                    child = list(dad)

                # MUTATION
                if self.mutation_type == 'swap':
                    child = self._mutate_swap(child)
                elif self.mutation_type == 'gaussian':
                    child = self._mutate_gaussian(child)
                else:
                    child = self._mutate_bit_flip(child)

                new_pop.append(child)

            self.pop = new_pop[:self.size]
            scr_list = [problem.evaluate(v) for v in self.pop]

            new_best_sol, new_best_scr = get_best_stat(self.pop, scr_list)
            diff = abs(new_best_scr - best_scr)
            best_sol, best_scr = new_best_sol, new_best_scr
            gen += 1

            yield {
                'generation':       gen,
                'current_solution': self.pop.copy(),
                'current_score':    scr_list.copy(),
                'best_solution':    best_sol,
                'best_score':       best_scr
            }

        return best_sol, best_scr"""