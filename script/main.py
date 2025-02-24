import numpy as np
from algorithm import *
from utils import *
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--function_name", type=str, default="ackley", choices=["cigar", "sphere", "ackley", "bohachevsky", "h1", "himmelblau", "rastrigin"])
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--F", type=float, default=0.8)
    parser.add_argument("--CR", type=float, default=0.9)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--algorithm", type=str, default="GA", choices=[ "GA", "DE", "PSO"])
    args = parser.parse_args()

    func = take_fitness_function(args.function_name)
    visualize_function(func)
    bounds = [[-5, 5], [-5, 5]]
    
    if args.algorithm == "GA":
        best_solution, best_value, history = GA(func=func, bounds=bounds,
                                                pop_size=args.pop_size, 
                                                CR=args.CR, 
                                                max_iter=args.max_iter)
    elif args.algorithm == "DE":
        best_solution, best_value, history = DE(func=func, bounds=bounds,
                                                pop_size=args.pop_size, 
                                                F=args.F,
                                                CR=args.CR, 
                                                max_iter=args.max_iter)
    
    elif args.algorithm == "PSO":
        best_solution, best_value, history = PSO(func=func, bounds=bounds,
                                                 pop_size=args.pop_size,
                                                 max_iter=args.max_iter)
                                                 
                                                 
    
    print("Best solution:", best_solution)
    print("Best value:", best_value)
    if args.visualize == True:
        ani = visualize_proccess(history, func)
        ani.save(f"../gif/{args.algorithm}_{args.function_name}.gif")
