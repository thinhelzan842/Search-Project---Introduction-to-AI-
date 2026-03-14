import numpy as np
import random
from core import AlgorithmBase

class GeneticAlgorithm(AlgorithmBase):
    def __init__(self, size, gen, desire=None, mutate_prob=0.1, crossover_prob=0.9,
                 tournament_size=3, crossover_type='one_point', mutation_type='bit_flip', 
                 sigma=0.1, points=2, tol=1e-6):
        self.size, self.gen, self.desire = size, gen, desire
        self.mutate_prob, self.crossover_prob = mutate_prob, crossover_prob
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.sigma, self.points = sigma, points
        self.tol = tol # Thêm dung sai (tolerance) để kiểm soát độ hội tụ chặt chẽ hơn

    def name(self) -> str: 
        return f"GA ({self.crossover_type} + {self.mutation_type})"

    # --- CROSSOVER STRATEGIES ---
    def _crossover_one_point(self, dad, mom):
        min_len = min(len(dad), len(mom))
        if min_len < 2: return list(mom)
        pt = random.randint(1, min_len - 1)
        return list(dad[:pt]) + list(mom[pt:])

    def _crossover_multi_point(self, dad, mom):
        min_len = min(len(dad), len(mom))
        if min_len < 2: return list(dad)
        max_pts = min_len - 1
        pts = sorted(random.sample(range(1, min_len), min(self.points, max_pts)))
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
            'generation': 0, 'current_solution': self.pop.copy(), 'current_score': scr_list.copy(),
            'best_solution': best_sol, 'best_score': best_scr
        }

        gen, diff = 0, float('inf')

        # CẬP NHẬT: Thay 0.1 bằng self.tol để tránh việc GA dừng quá sớm
        while (self.desire is None or best_scr > self.desire) and gen < self.gen and diff > self.tol:
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
                'generation': gen, 'current_solution': self.pop.copy(), 'current_score': scr_list.copy(),
                'best_solution': best_sol, 'best_score': best_scr
            }

        return best_sol, best_scr