import numpy as np
from typing import List, Tuple


def GA(func, bounds: List[Tuple[float, float]], pop_size: int, CR: float, max_iter: int):
    dim = len(bounds)
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    history = []
    
    for iter in range(max_iter):
        pool = []
        for _ in range(pop_size // 2):
            x1, x2 = pop[np.random.choice(pop_size, 2, replace=False)].copy()
            
            # Crossoverz
            j_rand = np.random.randint(0, dim)
            x1[j_rand], x2[j_rand] = x2[j_rand], x1[j_rand]
            
            # Mutation
            mutation_mask = np.random.rand(dim) < CR
            x1[mutation_mask] = np.random.uniform(
                [bounds[d][0] for d in range(dim)],
                [bounds[d][1] for d in range(dim)]
            )[mutation_mask]
            
            pool.append(x1)
            pool.append(x2)
        
        pool.extend(pop)
        pool = np.array(pool)
        
        # print(f"iteration {iter}: {pool}")
        # input()
        
        # Tournament Selection
        pop = tournament_selection(pool, func)
        
        history.append(pop)
    
    best_idx = np.argmin([func(indx) for indx in pop])
    best_solution = pop[best_idx]
    best_value = func(best_solution)
    
    return best_solution, best_value, history

def tournament_selection(pool, func):
    N = len(pool) // 2
    pop_selected = []
    pool_fitness = np.array([func(ind) for ind in pool])
    for i in range(2):
        idxs = np.arange(pool.shape[0])
        np.random.shuffle(idxs)   
        current_fitness = pool_fitness[idxs]
        current_pool = pool[idxs]
        for j in range(0, pool.shape[0], 4):
            idx = j + np.argmin(current_fitness[j : j + 4])
            pop_selected.append(current_pool[idx])
            
    pop_selected = np.array(pop_selected)   
    return pop_selected

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

def PSO(func, bounds: List[Tuple[float, float]], pop_size: int, max_iter: int, w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
    dim = len(bounds)
    pop = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim))
    vel = np.random.uniform(-1, 1, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    
    pbest = pop.copy() # population best
    pbest_fitness = fitness.copy()
    gbest_idx = np.argmin(fitness) 
    gbest = pop[gbest_idx] # partical best
    gbest_fitness = fitness[gbest_idx] # partical best fitness
    history = []
    
    for _ in range(max_iter):
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        vel = w * vel + c1 * r1 * (pbest - pop) + c2 * r2 * (gbest - pop) # create new
        
        # new swarm
        pop = np.clip(pop + vel, [b[0] for b in bounds], [b[1] for b in bounds])
        new_fitness = np.array([func(ind) for ind in pop])
        
        # update best swarm by the better canddidatess in new swarm
        mask = new_fitness < pbest_fitness
        mask = mask.reshape(-1,)
        pbest[mask] = pop[mask]
        pbest_fitness[mask] = new_fitness[mask]
        
        # update the best candidate
        if np.min(new_fitness) < gbest_fitness:
            gbest_idx = np.argmin(new_fitness)
            gbest = pop[gbest_idx]
            gbest_fitness = new_fitness[gbest_idx]
            
        history.append(pop.copy())
        
    return gbest, gbest_fitness, history

    
        
            
        


        
    
        