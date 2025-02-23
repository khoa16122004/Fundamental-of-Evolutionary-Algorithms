import numpy as np
from typing import List, Tuple
import numpy as np

def DE(func, bounds: List[Tuple[float, float]], pop_size: int, F: float, CR: float, max_iter: int):
    dim = len(bounds)
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    history = []
    
    for _ in range(max_iter):
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice([idx for idx in range(pop_size) if idx != i], size=3, replace=False) # not duplicated sample
            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            
            v = x1 + F * (x2 - x3)
            v = np.clip(v, [b[0] for b in bounds], [b[1] for b in bounds])
            
            j_rand = np.random.randint(0, dim)
            u = np.array([v[j] if np.random.rand() < CR or j == j_rand else pop[i][j] for j in range(dim)])
            
            new_fitness = func(u)
            if new_fitness < fitness[i]:
                pop[i] = u
                fitness[i] = new_fitness
                
        history.append(pop.copy())
        
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx]
    best_value = fitness[best_idx]
    return best_solution, best_value, history



        
    
        