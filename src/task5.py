import numpy as np
from numpy import *
import gurobipy as gp
from gurobipy import GRB
from data import data

# Sub problem: Solves the pricing poblem:
def solve_pricing_problem(V, cost, time, Q, demand, twstart, twend, lambdas):
    EPS = 0.0001

    N = V[1:-1]
    A = [(i,j) for i in V for j in V if i != j]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    ppm = gp.Model("Pricing Problem", env=env)
    ppm.setParam(gp.GRB.param.Presolve, 0)
    ppm.setParam(gp.GRB.param.Method, 0)

    ### MODEL START   -------------
    #Variables:
    x = ppm.addVars(A, vtype=GRB.BINARY, name="x")
    y = ppm.addVars(V, lb=0.0, ub=Q, vtype=GRB.CONTINUOUS, name="y")
    z = ppm.addVars(V, lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    #Objective function:
    #     when i = 0, the lambda_0 will be subtracted, so it is left out of this implementation to
    #     avoid subtracting it twice. This will calculate the correct reduced cost
    ppm.setObjective(gp.quicksum((cost[i][j] - lambdas[i-1]) * x[i,j] for i,j in A), GRB.MINIMIZE)

    #Flow constraints:
    for k in N:
        ppm.addConstr(gp.quicksum(x[i,k] for i,_ in A if i != k) == gp.quicksum(x[k,j] for _,j in A if j != k))

    ppm.addConstr(gp.quicksum(x[0,j] for j in N) == 1.0)
    ppm.addConstr(gp.quicksum(x[i,9] for i in N) == 1.0)

    # Capacity constraints:
    M1 = Q
    for i,j in A:
        if j != 0:
            ppm.addConstr(y[i] + demand[j] + (M1 * x[i,j]) - M1 <= y[j] + EPS)

    #Time window constraints:
    M2 = 24.0
    for i,j in A:
        ppm.addConstr(z[i] + time[i][j] - M2 * (1.0 - x[i,j]) <= z[j] + EPS)

    for i in V:
        ppm.addConstr(z[i] >= twstart[i])
        ppm.addConstr(z[i] <= twend[i])
    ### MODEL END   -------------

    # Solve
    ppm.update()
    # ppm.write("model.lp")
    ppm.optimize()

    # Find route with reduced cost
    route = np.array([0 for i in N])
    if ppm.status == GRB.OPTIMAL:
        routeCost = 0.0
        for var in ppm.getVars():
            if var.xn > 0:
                if "x" in var.VarName:
                    i = int(var.varName[2])
                    j = int(var.varName[4])
                    routeCost += cost[i][j]
                    if i > 0:
                        route[i-1] = 1
        return ppm.objVal, route, routeCost
    else:
        return 1,[], 0


# RMP solves the restricted master problem:
def solve_master_problem_with_cg(routes, costRoutes, customers, cost, time, cap, vehicles, demand, twstart, twend):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    m = gp.Model("MP", env=env)
    # Variables + objective:
    theta = m.addVars(routes.shape[0], obj=costRoutes, lb=0.0, name="theta", vtype=GRB.CONTINUOUS)

    #Constraints:
    for i in range(customers):
        m.addConstr(gp.quicksum((routes[p][i] * theta[p]) for p in range(len(routes))) == 1.0) # Visits

    m.addConstr(gp.quicksum(theta[p] for p in range(len(routes))) <= vehicles)  #Vehicle

    # Column Generation:
    j = 0
    while True:
        m.update()
        m.optimize()

        j += 1
        zpp, route, routeCost = solve_pricing_problem(range(customers+2), cost, time, cap, demand, twstart, twend, m.Pi)

        #Check if no new columns can be added:
        if(zpp >= 0): # or route_exists(routes, route)):
            print("No new columns could be added to the problem\n\n")
            break
        print(" Added route: ", route, "  - with cost: ", routeCost, " - red.cost: ", zpp)#, " pi: ", m.Pi)

        # Add to the list of routes and cost list:
        K = len(routes)
        routes = np.append(routes, [route], axis=0)
        costRoutes = np.append(costRoutes, [routeCost], axis=0)

        # Add variable:
        col = gp.Column()
        for i in range(len(m.getConstrs())):
            if m.getConstrs()[i].sense == "=":
                col.addTerms(route[i], m.getConstrs()[i])
            else:
                col.addTerms(1.0, m.getConstrs()[i])
        m.addVar(obj=routeCost, vtype=GRB.CONTINUOUS, column=col, name="theta[%i]" % K)

    #Returns objective value, routes and routes cost
    if m.status == GRB.OPTIMAL:
        for var in m.getVars():
            if "theta" in var.VarName:
                print(var)
        return m.objVal, routes, costRoutes
    return 0

# Check if route already exists in the list of routes
def route_exists(routes, route):
    K = len(routes)
    res = 0
    for r in routes:
        for j in range(len(r)):
            if r[j] != route[j]:
                res += 1
                break
    return K != res

if __name__ == "__main__":
    sol, r, c = solve_master_problem_with_cg(data.routes, data.costRoutes, data.cust, data.cost, data.timeCost, data.Q, data.m, data.demand, data.twStart, data.twEnd)
    print("ROUTES:\n", r)
    print("COST OF ROUTES: ", c)
    print("OBJECTIVE FUNCTION VALUE: ", sol)
