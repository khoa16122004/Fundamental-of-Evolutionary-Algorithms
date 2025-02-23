import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from deap import benchmarks # benchmark library
import matplotlib.animation as animation
from scipy.optimize import rosen
from IPython.display import HTML
from algorithm import DE

def take_fitness_function(benchmark_name: str):
    if hasattr(benchmarks, benchmark_name):
        return getattr(benchmarks, benchmark_name)
    else:
        raise ValueError("No benchmark named %s" % benchmark_name)

def take_algorithm(algorithm_name: str):
    if algorithm_name == "DE":
        return DE


def visualize_proccess(history, func):
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes 
    contour = ax1.contour(X, Y, Z, levels=30, cmap="jet")
    scat1 = ax1.scatter([], [], c='red', marker='o')
    ax1.set_title("Evolution on countour Plot")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)

    scat2 = ax2.scatter([], [], c='black', marker='.')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_title("Difference Vector Distribution")
  
    def update(frame):
        pop = history[frame]
        scat1.set_offsets(pop)
        diff_vectors = pop - np.mean(pop, axis=0)
        scat2.set_offsets(diff_vectors)
        return scat1, scat2

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100, repeat=True)
    plt.show()

    # return ani

def visualize_function(func):
  X = np.arange(-5, 5, 0.1)
  Y = np.arange(-5, 5, 0.1)
  X, Y = np.meshgrid(X, Y)
  Z = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      Z[i, j] = func([X[i, j], Y[i, j]])[0]

  fig = plt.figure(figsize=(15, 5))
  ax1 = fig.add_subplot(121, projection='3d')
  ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.1)
  ax1.set_title("3D Surface Plot")

  ax2 = fig.add_subplot(122)
  contour = ax2.contour(X, Y, Z, cmap=cm.jet, levels=30)
  fig.colorbar(contour, ax=ax2)
  ax2.set_title("Contour Plot")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()