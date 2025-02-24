# Fundamental-of-Evolutionary-Computation
This repo is a simple implementation of Genetic Algorithms (GA), Difference Evolution (DE) and Particle Swarm Optimzation (PSO) in the problem of minimizing some benchmark function.

# Usage 
Usage: python script.py [--function_name FUNCTION] [--pop_size POP_SIZE] [--F F_VALUE] 
                        [--CR CR_VALUE] [--max_iter MAX_ITER] [--visualize VISUALIZE] 
                        [--algorithm ALGORITHM]

Arguments:
  --function_name FUNCTION   The benchmark function to optimize. Choices: {"cigar", "sphere", 
                             "ackley", "bohachevsky", "h1", "himmelblau", "rastrigin"}.
                             Default: "ackley".
  --pop_size POP_SIZE        Population size (default: 50).
  --F F_VALUE                Scaling factor for differential evolution (default: 0.8).
  --CR CR_VALUE              Crossover probability (default: 0.9).
  --max_iter MAX_ITER        Maximum number of iterations (default: 100).
  --visualize VISUALIZE      Whether to visualize the optimization process (default: True).
  --algorithm ALGORITHM      Optimization algorithm to use. Choices: {"GA", "DE", "PSO"}.
                             Default: "GA".

Example usage:
  python script.py --function_name sphere --pop_size 100 --algorithm DE
  python script.py --F 0.5 --CR 0.7 --max_iter 200 --visualize False

