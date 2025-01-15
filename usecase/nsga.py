import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination

# Define the problem dimensions
N = 10
num_ins = 100  # Example number of instances

# Example data (replace with actual data)
price_all_loc = np.random.rand(N*N, num_ins)
water_all_loc = np.random.rand(N*N, num_ins)
carbon_all_loc = np.random.rand(N*N, num_ins)
workload_trace = np.random.rand(N, num_ins)
mask_array = np.ones((N*N, num_ins))
l_0 = 1
l_1 = 100
l_2 = 100
max_cap = 1

# Define the objective function
def evaluate(individual):
    x = np.array(individual).reshape((N*N, num_ins))
    x_masked = np.multiply(x, mask_array)
    
    energy_cost = np.sum(np.multiply(x_masked, price_all_loc))
    
    water_cost = np.sum(np.multiply(x_masked, water_all_loc), axis=1)
    water_cost = np.reshape(water_cost, (N, N))
    water_cost = np.sum(water_cost, axis=1)
    
    carbon_cost = np.sum(np.multiply(x_masked, carbon_all_loc), axis=1)
    carbon_cost = np.reshape(carbon_cost, (N, N))
    carbon_cost = np.sum(carbon_cost, axis=1)
    
    total_cost = l_0 * energy_cost + l_1 * np.max(water_cost) + l_2 * np.max(carbon_cost)
    
    return [total_cost]

# Define the constraints
def feasible(individual):
    x = np.array(individual).reshape((N*N, num_ins))
    x_masked = np.multiply(x, mask_array)
    
    for i in range(num_ins):
        for j in range(N):
            if np.sum(x_masked[N*j:N*j+N, i]) > max_cap:
                return False
            if np.sum(x_masked[j::N, i]) != workload_trace[j, i]:
                return [False]
    return [True]

# Define the problem
class ResourceAllocationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=N*N*num_ins, n_obj=1, n_constr=1, xl=0, xu=1)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate(x)
        out["G"] = feasible(x)

# Create the NSGA-II algorithm instance
algorithm = NSGA2(
    pop_size=300,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PolynomialMutation(prob=0.1, eta=20),
    eliminate_duplicates=True
)

# Define the problem instance
problem = ResourceAllocationProblem()

# Define the termination criterion
termination = get_termination("n_gen", 50)

# Run the algorithm
res = minimize(problem, algorithm, termination, seed=1, verbose=True)

# Get the best individual
best_individual = res.X[np.argmin(res.F)]
print("Best individual is: ", best_individual)
print("With fitness: ", res.F[np.argmin(res.F)])
