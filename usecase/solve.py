import numpy as np 
import cvxpy as cp
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


def offline_solver(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, mask_array, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10*10, num_ins]
        water_all_loc   : Water WUE of all locations [10*10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10*10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 10, num_ins]
        num_ins         : Number of timesteps to solve
        l_1             : Water coeficiency
        l_2             : Carbon coeficiency
    Return:
        optimal_cost:
        action_mask:
    '''
    x            = cp.Variable([10*10, num_ins])
    x_masked     = cp.multiply(x, mask_array)
    
    energy_cost  = cp.sum(cp.multiply(x_masked, price_all_loc))

    water_cost   = cp.sum(cp.multiply(x_masked, water_all_loc), axis=1)
    water_cost   = cp.reshape(water_cost, [10,10], order="C")
    water_cost   = cp.sum(water_cost, axis=1)

    carbon_cost  = cp.sum(cp.multiply(x_masked, carbon_all_loc), axis=1)
    carbon_cost  = cp.reshape(carbon_cost, [10,10], order="C")
    carbon_cost  = cp.sum(carbon_cost, axis=1)

    
    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    for i in range(num_ins):
        for j in range(10):
            c_i = [
                cp.sum(x_masked[10*j:10*j+10, i]) <= max_cap,
                cp.sum(x_masked[j::10, i]) ==  workload_trace[j,i]
            ]
            constraints += c_i

    for i in range(num_ins):
        for j in range(100):
            constraints += [x[j,i] >= 0]


    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(solver=cp.GUROBI,verbose = verbose)
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return optimal_cost, action_mask


def offline_solver_capa(price_all_loc, water_all_loc, carbon_all_loc, capacity_all_loc,
                   workload_trace, mask_array, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100,l_3=100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10*10, num_ins]
        water_all_loc   : Water WUE of all locations [10*10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10*10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 10, num_ins]
        num_ins         : Number of timesteps to solve
        l_1             : Water coeficiency
        l_2             : Carbon coeficiency
    Return:
        optimal_cost:
        action_mask:
    '''
    x            = cp.Variable([10*10, num_ins])
    x_masked     = cp.multiply(x, mask_array)
    
    energy_cost  = cp.sum(cp.multiply(x_masked, price_all_loc))

    water_cost   = cp.sum(cp.multiply(x_masked, water_all_loc), axis=1)
    water_cost   = cp.reshape(water_cost, [10,10], order="C")
    water_cost   = cp.sum(water_cost, axis=1)

    carbon_cost  = cp.sum(cp.multiply(x_masked, carbon_all_loc), axis=1)
    carbon_cost  = cp.reshape(carbon_cost, [10,10], order="C")
    carbon_cost  = cp.sum(carbon_cost, axis=1)

    capacity_cost = cp.sum(cp.multiply(x_masked, capacity_all_loc), axis=1)
    capacity_cost = cp.reshape(capacity_cost, [10,10], order="C")
    capacity_cost = cp.sum(capacity_cost, axis=1)
    
    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
        capacity_cost = cp.norm(capacity_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
        capacity_cost = cp.sum(capacity_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost + l_3*capacity_cost

    constraints = []
    for i in range(num_ins):
        for j in range(10):
            c_i = [
                cp.sum(x_masked[10*j:10*j+10, i]) <= max_cap,
                cp.sum(x_masked[j::10, i]) ==  workload_trace[j,i]
            ]
            constraints += c_i

    for i in range(num_ins):
        for j in range(100):
            constraints += [x[j,i] >= 0]


    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI())
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return optimal_cost, action_mask


def solve_action(virtual_price, workload, max_cap, mask_array, num_nodes = 10):
    '''
    Solve the reward function problem arg min x∈Xt {⟨pt, xt⟩ + γt · bt · xt} 
    Args:
        virtual_price   : Energy price of all locations [num_ins, num_ins]
        workload        : Workload trace
        max_cap         : The maximum capability of datacenter
        mask_array      : Array with size of [num_ins, num_ins]
        num_nodes       : Number of datacenter
    Return:
        action_mask     : Action based on mask
        optimal_cost    : Optimal cost 
    '''
    assert virtual_price.shape == (num_nodes, num_nodes)
    
    x            = cp.Variable([num_nodes, num_nodes])
    x_masked     = cp.multiply(x, mask_array)
    
    constraints = []
    for i in range(10):
        c_i = [
            cp.sum(x_masked[i, :]) <= max_cap,
            cp.sum(x_masked[:, i]) == workload[i]
        ]
        
        constraints += c_i
    constraints += [x_masked >= 0]
    
    total_cost   = cp.sum(cp.multiply(x_masked, virtual_price))
    objective    = cp.Minimize(total_cost)
    prob         = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI(),verbose = False)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return action_mask, optimal_cost

def solve_auxiliary(vector_gamma, z_max_array, num_nodes = 10):
    '''
    Solve the reward problem
    Args:
        vector_gamma    : Vector of gamma_t
        z_max_array     : the maximum value of z
        num_nodes       : Number of datacenter
    Return:
        z_optimal       : Optimal auxiliary action
        optimal_cost    : Optimal cost 
    '''
    
    assert vector_gamma.shape == (num_nodes*2,)
    
    z = cp.Variable([2*num_nodes])
    
    constraints  = []
    constraints += [z >= 0]
    constraints += [z <= z_max_array]
    
    total_cost    = cp.atoms.norm(z[:num_nodes], "inf") + cp.atoms.norm(z[num_nodes:], "inf") \
                     - vector_gamma @ z 
    objective     = cp.Minimize(total_cost)
    prob          = cp.Problem(objective, constraints)
    
    prob.solve(verbose = False)
    
    optimal_cost  = prob.value
    z_optimal     = z.value
    
    
    return z_optimal, optimal_cost



