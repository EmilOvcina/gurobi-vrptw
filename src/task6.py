import numpy as np
from numpy import *
import gurobipy as gp
from gurobipy import GRB
from data import data

# Sub problem: Solves the pricing poblem:
def solve_pricing_problem(V, cost, time, Q, demand, twstart, twend, pi, mu):
    EPS = 0.0001
    N = V[1:-1]
    A = [(i,j) for i in V for j in V if i != j]

    lambdas = pi[:len(N)+1]

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
    w = ppm.addVars(mu, vtype=GRB.BINARY, name="w")

    #Objective function:
    #     when i = 0, the lambda_0 will be subtracted, so it is left out of this implementation to
    #     avoid subtracting it twice. This will calculate the correct reduced cost
    ppm.setObjective(gp.quicksum((cost[i][j] - lambdas[i-1]) * x[i,j] for i,j in A) \
        - gp.quicksum(mu[i,j,k] * w[i,j,k] for i,j,k in mu), GRB.MINIMIZE)

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
        ppm.addConstr(z[i] <= twend[i] + EPS)

    #SR cut constraints:
    ppm.addConstrs((gp.quicksum(x[i,s] for s in V if s != i) + gp.quicksum(x[j,s] for s in V if s != j)\
        + gp.quicksum(x[k, s] for s in V if s != k) - 1 <= w[i,j,k] * 2) for i,j,k in mu)
    ### MODEL END   -------------

    # Solve
    ppm.update()
    ppm.optimize()

    # Find route with reduced cost
    route = np.array([0 for i in N])
    if ppm.status == GRB.OPTIMAL:
        xss = []
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
        m.addConstr(gp.quicksum((routes[p][i] * theta[p]) for p in range(len(routes))) == 1.0) # Visist

    m.addConstr(gp.quicksum(theta[p] for p in range(len(routes))) <= vehicles)  #Vehicle

    # SR cuts loop:
    w = 0
    vio_cust = []
    mu = {}
    while True:
        w+= 1

        # Column Generation:
        j = 0
        while True:
            m.update()
            m.optimize()

            if len(vio_cust) > 0: # Adds the SR cut to a dict used by the pricing problem
                mu[vio_cust[0],vio_cust[1],vio_cust[2]] = m.Pi[-1]

            j += 1
            zpp, route, routeCost = solve_pricing_problem(range(customers+2), cost, time, cap, demand, twstart, twend, m.Pi, mu)

            #Check if no new columns can be added:
            if(zpp >= 0 or route_exists(routes, route)):
                print("No more columns to add")
                break
            print("Added column: ", route, " - cost: ", routeCost, " - red.cost: ", zpp)

            # Add to the list of routes and cost list:
            K = len(routes)
            routes = np.append(routes, [route], axis=0)
            costRoutes = np.append(costRoutes, [routeCost], axis=0)

            # Add variable:
            col = gp.Column()
            for i in range(len(m.getConstrs())):
                if m.getConstrs()[i].sense == "=":
                    col.addTerms(route[i], m.getConstrs()[i])
                elif i == 7:
                    col.addTerms(1.0, m.getConstrs()[i])
                elif i > 7:
                    col.addTerms(0.0, m.getConstrs()[i])
            m.addVar(obj=routeCost, vtype=GRB.CONTINUOUS, column=col, name="theta[%i]" % K)

        ## Check cuts here
        ts = []
        for var in m.getVars():
            if "theta" in var.VarName:
                ts.append(var.x)

        vio_routes, vio_cust = find_sr_cut(ts, routes.T, customers)

        #If no more cuts or integrality constraints are satisfied, break
        if (vio_cust[0],vio_cust[1],vio_cust[2]) in mu or check_for_integrality(ts):
            print("No more cuts to add!\n")
            break

        # Create and add new constraint:
        print("Added SR-cut: ", vio_routes, " for ", vio_cust)
        br = create_constr_for_sr(vio_routes, len(routes))
        m.addConstr(gp.quicksum(br[i] * m.getVars()[i] for i in range(len(br))) <= 1)

    #Returns objective value, routes and routes cost
    if m.status == GRB.OPTIMAL:
        selected_routes = []
        for var in m.getVars():
            if "theta" in var.VarName:
                print(var)
                if var.x > 0.0:
                    selected_routes.append(var.VarName)
        return m.objVal, routes, costRoutes, selected_routes
    else:
        return 0, [], [], []

# Check if any entry in thetas is not integral
def check_for_integrality(thetas):
    for t in thetas:
        if t != 0.0 and t != 1.0:
            return False
    return True

# Makes a row to insert as a constraint in the model
def create_constr_for_sr(vio_routes, length):
    res = np.zeros(length)
    for i in vio_routes:
        res[i] = 1
    return res

# Finds the most violating SR cut:
def find_sr_cut(theta, rows, customers):
    most_violated = 1.0
    most_vio_routes = []
    vio_cust = []
    for i in range(customers):
        for j in range(i+1, customers):
            for k in range(j+1, customers):
                violation = 0.0
                vio_routes = []
                for p in range(len(theta)): # Go to each entry for all triplets of customers
                    if rows[i][p] + rows[j][p] + rows[k][p] >= 2: #Check if column p is violating
                       violation += theta[p]
                       vio_routes.append(p)
                if violation >= most_violated: #if violation is worse than previous, set new most_violated and update routes
                    most_violated = violation
                    most_vio_routes = vio_routes
                    vio_cust = [i,j,k]
    return most_vio_routes, vio_cust

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
    # Routes from task 4:
    routes = np.array([
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
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        ])
    costRoutes = np.array([15., 12., 22., 18., 15., 22., 18., 10., 15., 11., 13., 12., 11., 18., 15.,  6.])
    sol, r, c, sel_routes = solve_master_problem_with_cg(routes, costRoutes, data.cust, data.cost, data.timeCost, data.Q, data.m, data.demand, data.twStart, data.twEnd)
    # sol, r, c, sel_routes = solve_master_problem_with_cg(data.routes, data.costRoutes, data.cust, data.cost, data.timeCost, data.Q, data.m, data.demand, data.twStart, data.twEnd)
    print("ROUTES: ", r)
    print("COST OF ROUTES: ", c)
    print("SELECTED ROUTES: ", sel_routes)
    print("OBJECTIVE FUNCTION VALUE: ", sol)
