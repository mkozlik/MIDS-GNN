import networkx as nx
import numpy as np
import gurobipy as gp
import time


def optimize_Gurobi(G, problem, goal="D", outputFlag=1, single_cpu=False):
    A = nx.to_numpy_array(G)
    n = np.size(A, 1)
    A_ = A + np.eye(n)

    # Set up environment.
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", outputFlag)
    env.start()

    # Create a new model.
    model = gp.Model("MIDS", env=env)
    model.setParam("presolve", 0)
    if single_cpu:
        model.setParam('Threads', 1)

    # Add variables to the model.
    D = model.addVars(G.nodes, vtype=gp.GRB.BINARY)

    # Add constraints.
    if problem == "MDS" or problem == "MIDS":
        # Constraint for domination: (A+I)*D >= 1 for all nodes
        con1 = model.addConstrs(
            (sum([A_[i, j] * D[k] for j, k in enumerate(G.nodes)]) >= 1 for i in range(n)), name="Con1" # type: ignore
        )
    if problem == "MIS" or problem == "MIDS":
        # Constraint for independence: D[i] + D[j] <= 1 for all edges (i, j)
        con2 = model.addConstrs((D[edge[0]] + D[edge[1]] <= 1 for edge in G.edges), name="Con2")

    # Set the objective
    if problem == "MDS" or problem == "MIDS":
        if goal == "D":
            model.setObjective(D.sum(), gp.GRB.MINIMIZE)
        elif goal == "J":
            model.setObjective(
                sum(sum(A_[i, j] * D[k] for j, k in enumerate(G.nodes)) for i in range(n)), gp.GRB.MINIMIZE
            )
    else:
        model.setObjective(D.sum(), gp.GRB.MAXIMIZE)

    # Start the optimization.
    start = time.perf_counter()
    model.optimize()
    end = time.perf_counter()

    solution = []
    if model.getAttr("SolCount") >= 1:
        for vertex in G.nodes:
            if D[vertex].X > 0.5:
                solution.append(vertex)

    details = dict(
        goal=goal,
        goal_value=model.getAttr("ObjVal"),
        lenD=len(solution),
        valJ=sum(sum(A_[i, j] * D[k].X for j, k in enumerate(G.nodes)) for i in range(n)),
        #    A=A,
        d=solution,
    )

    return solution, (end - start) * 1000, details