import numpy as np 
import gurobipy as gp

def offline_solver_gurobi(price_all_loc, water_all_loc, carbon_all_loc, workload_trace, timesteps, l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, verbose=True, f_type="AVG"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10, timesteps]
        water_all_loc   : Water WUE of all locations [10, timesteps]
        carbon_all_loc  : Carbon consumption of all places [10, timesteps]
        workload_trace  : Workload trace
        timesteps       : Number of timesteps to solve
        l_0             : Cost coeficiency
        l_1             : Water coeficiency
        l_2             : Carbon coeficiency
    Return:
        optimal_cost:
        energy_usage: Energy usage per Data Center [10, timesteps]
    '''
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            x = m.addMVar((10, timesteps), vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=max_cap)
            max_total_water = m.addVar(name="max_total_water")
            max_total_carbon = m.addVar(name="max_total_carbon")
            energy_cost = sum(sum(x[i] * price_all_loc[i]) for i in range(10))
            if f_type == "AVG":
                water_cost = sum(sum((x[i]+0.33)*1.1 * water_all_loc[i]) for i in range(10))
                carbon_cost = sum(sum((x[i]+0.33)*1.1 * carbon_all_loc[i]) for i in range(10))
            elif f_type == "MAX":
                for i in range(10):
                    m.addConstr(sum((x[i]+0.33)*1.1 * water_all_loc[i]) <= max_total_water)
                    m.addConstr(sum((x[i]+0.33)*1.1 * carbon_all_loc[i]) <= max_total_carbon)
                water_cost = max_total_water
                carbon_cost = max_total_carbon
            else:
                raise NotImplementedError
            m.update()
            # constraints
            for i in range(timesteps):
                m.addConstr(x[:, i].sum()  == workload_trace[i])
            m.setObjective(l_0 * energy_cost + l_1 * water_cost + l_2 * carbon_cost, gp.GRB.MINIMIZE)
            m.optimize()
            optimal_cost = m.objVal
            energy_usage = x.X
            return optimal_cost, (energy_usage+0.33)*1.1


def offline_solver_gurobi_capa(price_all_loc, water_all_loc, carbon_all_loc, capa_all_loc, workload_trace, timesteps, l_0 = 1, l_1 = 100, l_2 = 100, l_3=100, max_cap = 1, verbose=True, f_type="AVG"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10, timesteps]
        water_all_loc   : Water WUE of all locations [10, timesteps]
        carbon_all_loc  : Carbon consumption of all places [10, timesteps]
        capacity_all_loc: Capacity of all locations [10, timesteps]
        workload_trace  : Workload trace
        timesteps       : Number of timesteps to solve
        l_0             : Cost coeficiency
        l_1             : Water coeficiency
        l_2             : Carbon coeficiency
    Return:
        optimal_cost:
        energy_usage: Energy usage per Data Center [10, timesteps]
    '''

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            x = m.addMVar((10, timesteps), vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=max_cap)
            energy_cost = sum(sum(x[i] * price_all_loc[i]) for i in range(10))
            if f_type == "AVG":
                water_cost = sum(sum(x[i] * water_all_loc[i]) for i in range(10))
                capacity_cost= sum(sum(x[i] *capa_all_loc[i]) for i in range(10))
                carbon_cost = sum(sum(x[i] * carbon_all_loc[i]) for i in range(10))
            elif f_type == "MAX":
                water_cost = max(sum(x[i] * water_all_loc[i]) for i in range(10))
                capacity_cost= max(sum(x[i] *capa_all_loc[i]) for i in range(10))
                carbon_cost =max(sum(x[i] * carbon_all_loc[i]) for i in range(10))
            else:
                raise NotImplementedError
            m.update()
            # constraints
            for i in range(timesteps):
                m.addConstr(x[:, i].sum()  == workload_trace[i])
            m.setObjective(l_0 * energy_cost + l_1 * water_cost + l_2 * carbon_cost + l_3*capacity_cost, gp.GRB.MINIMIZE)
            m.optimize()
            optimal_cost = m.objVal
            energy_usage = x.X
            return optimal_cost, (energy_usage+0.33)*1.1
