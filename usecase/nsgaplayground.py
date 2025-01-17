from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

class MyConstrainedProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=3, n_obj=3, n_constr=1, xl=-5, xu=5)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2
        f2 = (x[1] - 2)**2
        f3 = (x[2] + 1)**2
        out["F"] = [f1, f2, f3]
        
        # Constraint: sum of decision variables must be <= 1
        g1 = np.sum(x) - 1
        out["G"] = [g1]

problem = MyConstrainedProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()