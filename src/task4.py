import numpy as np
from numpy import *
import gurobipy as gp
from gurobipy import GRB
from data import data

# Solves the (restricted) master problem and print dual variables
def rmp(customers, vehicles, routes, cost_routes):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("RMP", env=env)
    # Variables + objective:
    theta = m.addVars(routes.shape[0], obj=cost_routes, lb=0.0, name="theta", vtype=GRB.CONTINUOUS)

    #Constraints:
    for i in range(customers):
        m.addConstr(gp.quicksum((routes[p][i] * theta[p]) for p in range(len(routes))) == 1.0) # Visits

    m.addConstr(gp.quicksum(theta[p] for p in range(len(routes))) <= vehicles)  #Vehicle

    m.update()
    m.optimize()

    if m.status == GRB.OPTIMAL:
        for var in m.getVars():
            if "theta" in var.VarName:
                print(var)
        print(m.objVal)
    return m.Pi

if __name__ == "__main__":
    #The initial routes for Task 4
    #Each row describe the customers visited in a route. If the n'th index in a row is '1.0', then the route visits customer n.
    routes = array([
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1]
    ])

    #The distance cost of the initial routes.
    cost = array([15.0, 12.0, 22.0, 18.0, 15.0, 22.0, 18.0, 10.0, 15.0, 11.0, 13.0, 12.0])

    customers = data.cust
    vehicles = data.m

    #initial:
    lambdas = rmp(customers, vehicles, routes, cost)

    # # 1st iteration:
    # Insert a column with negative cost:
    new_route = [0,0,0,0,1,1,1,0]
    routes = np.append(routes, [new_route], axis=0)
    # Insert cost of the new route:
    cost = np.append(cost, [11.0])
    # Resolve the restricted master problem
    lambdas = rmp(customers, vehicles, routes, cost)

    # # 2nd iteration:
    # Insert a column with negative cost:
    new_route = [1,1,1,0,0,0,0,0]
    routes = np.append(routes, [new_route], axis=0)
    # Insert cost of the new route:
    cost = np.append(cost, [18.0])
    # Resolve the restricted master problem
    lambdas = rmp(customers, vehicles, routes, cost)

    # # 3rd iteration:
    # Insert a column with negative cost:
    new_route = [1,1,0,0,0,0,0,0]
    routes = np.append(routes, [new_route], axis=0)
    # Insert cost of the new route:
    cost = np.append(cost, [15.0])
    # Resolve the restricted master problem
    lambdas = rmp(customers, vehicles, routes, cost)

    # # 4th iteration:
    # Insert a column with negative cost:
    new_route = [0,0,0,1,0,0,0,0]
    routes = np.append(routes, [new_route], axis=0)
    # Insert cost of the new route:
    cost = np.append(cost, [6.0])
    # Resolve the restricted master problem
    lambdas = rmp(customers, vehicles, routes, cost)


#Used for calculating the reduced cost - only used when feasible routes have been found
def find_red_cost(lit_route, lambdas, cost_matrix):
    lambdai = lambdas[:-1]
    lambda0 = lambdas[-1]
    result = 0
    for i in range(len(lit_route) - 1):
        j = i+1
        if lit_route[i] > 0:
            result += (cost_matrix[lit_route[i]][lit_route[j]] - lambdai[lit_route[i] - 1]) - lambda0
        else:
            result += (cost_matrix[lit_route[i]][lit_route[j]])
    return result
