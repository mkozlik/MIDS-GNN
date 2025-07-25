import random
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import gurobipy as gp
import time
import math


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



# ----------------------------------------------------------------------------------
# Utility functions for independent dominating set checking and solution extension.
# ----------------------------------------------------------------------------------

def is_independent(G, S):
    """Return True if S is an independent set in graph G."""
    for u in S:
        for v in S:
            if u != v and G.has_edge(u, v):
                return False
    return True

def is_dominating(G, S):
    """Return True if S is a dominating set of G
       (every vertex not in S has at least one neighbor in S)."""
    for v in G.nodes():
        if v not in S:
            # if no neighbor of v is in S then S is not dominating
            if not any(u in S for u in G.neighbors(v)):
                return False
    return True

def is_maximal_independent(G, S):
    """Return True if S is an independent and dominating set (i.e. a maximal independent set)."""
    return is_independent(G, S) and is_dominating(G, S)

def extend_to_maximal(G, S):
    """
    Given an independent set S, extend it greedily so that it becomes maximal.
    (That is, add free vertices that are currently not dominated by S.)
    """
    S = set(S)
    # Add any vertex that is not dominated (has no neighbor in S) until the set is dominating.
    added = True
    while added:
        added = False
        for v in G.nodes():
            if v not in S:
                if not any(n in S for n in G.neighbors(v)):
                    S.add(v)
                    added = True
    return S

# ----------------------------------------------------------------------------------
# Helper functions for computing "tightness" and building an initial solution.
# ----------------------------------------------------------------------------------

def compute_tightness(G, S):
    """
    For every vertex v not in S, compute its tightness,
    i.e. the number of neighbors that belong to S.
    """
    tightness = {}
    for v in G.nodes():
        if v not in S:
            tightness[v] = sum(1 for u in G.neighbors(v) if u in S)
    return tightness

def greedy_initial_solution(G):
    """
    Build a greedy initial solution.
    Starting from an empty set, pick vertices (e.g., in order of decreasing degree)
    so that when a vertex is added, it “covers” itself and its neighbors.
    """
    S = set()
    uncovered = set(G.nodes())
    # In each iteration, take a vertex of maximum degree from the uncovered set.
    while uncovered:
        v = max(uncovered, key=lambda x: G.degree(x))
        S.add(v)
        # Remove v and its neighbors from uncovered vertices.
        uncovered.discard(v)
        for u in G.neighbors(v):
            uncovered.discard(u)
    # Extend S so that it is maximal (it will be independent by construction)
    S = extend_to_maximal(G, S)
    return S

# ----------------------------------------------------------------------------------
# Local Search and Plateau Search Moves (using k-swap ideas)
# ----------------------------------------------------------------------------------

def local_search(S, G, k=2, max_iter=100):
    """
    Perform a local search that seeks an improvement using a k-swap.

    For example, for k = 2 the code checks every non-solution vertex v with tightness 2.
    Let D be the two solution neighbors of v.
    If (S \ D) ∪ {v} can be extended to a valid (and smaller) maximal independent set,
    the move is accepted.
    """
    S = set(S)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        iter_count += 1
        improved = False
        tightness = compute_tightness(G, S)
        candidate_found = None
        # Look for a vertex with tightness exactly equal to k.
        for v, t in tightness.items():
            if t == k:
                # Get the k neighbors of v that belong to S.
                D = {u for u in G.neighbors(v) if u in S}
                # Candidate: swap these k vertices for v.
                S_candidate = (S - D) | {v}
                # Extend candidate to be maximal.
                S_candidate = extend_to_maximal(G, S_candidate)
                # Accept the candidate if it improves the solution size and is valid.
                if len(S_candidate) < len(S) and is_maximal_independent(G, S_candidate):
                    candidate_found = S_candidate
                    break
        if candidate_found is not None:
            S = candidate_found
            improved = True
    return S

def plateau_search(S, G):
    """
    Perform a plateau search which seeks to move on the plateau.

    For each non-solution vertex v that is 1-tight (i.e. exactly one neighbor x in S),
    try swapping x and v. Then, extend to a maximal independent set.
    """
    S = set(S)
    tightness = compute_tightness(G, S)
    candidate_found = None
    for v, t in tightness.items():
        if t == 1:
            # Find the unique neighbor x of v in S.
            neighbors_in_S = [u for u in G.neighbors(v) if u in S]
            if len(neighbors_in_S) == 1:
                x = neighbors_in_S[0]
                S_candidate = (S - {x}) | {v}
                # It must remain independent.
                if is_independent(G, S_candidate):
                    S_candidate = extend_to_maximal(G, S_candidate)
                    if is_maximal_independent(G, S_candidate):
                        candidate_found = S_candidate
                        break
    return candidate_found if candidate_found is not None else S

# ----------------------------------------------------------------------------------
# Vertex Penalty and Kick (perturbation) functions
# ----------------------------------------------------------------------------------

def update_penalty(penalty, S, delta):
    """
    Increase the penalty for each vertex in the current solution S.
    Then (optionally) reduce penalty scores to "forget" old history.
    """
    for v in S:
        penalty[v] = penalty.get(v, 0) + 1
    # Simple decay step: limit the penalty values by delta and halve them.
    for v in penalty:
        penalty[v] = int(min(penalty[v], delta) / 2)
    return penalty

def kick(S_star, penalty, G, nu):
    """
    Generate a new initial solution by "kicking" the incumbent S_star.
    A set R of non-solution vertices is selected iteratively.
    In each trial, pick a vertex (from V \ (S* U neighbors(R))) with the smallest penalty.
    With probability 1/nu, stop; otherwise try adding another vertex.
    Then, produce a new solution S = (S_star - N(R)) U R and extend it.
    """
    R = set()
    S_star = set(S_star)
    all_nodes = set(G.nodes())
    while True:
        forbidden = S_star | set().union(*(set(G.neighbors(v)) for v in R))
        candidates = list(all_nodes - forbidden)
        if not candidates:
            break
        # Pick the candidate with the lowest penalty.
        candidate = min(candidates, key=lambda v: penalty.get(v, 0))
        R.add(candidate)
        # Stop with probability 1/nu.
        if random.random() < 1.0 / nu:
            break
    # Remove from S_star all vertices adjacent to R and add R.
    N_R = set()
    for v in R:
        N_R.update(G.neighbors(v))
    S_new = (S_star - N_R) | R
    S_new = extend_to_maximal(G, S_new)
    return S_new

# ----------------------------------------------------------------------------------
# Iterated Local & Plateau Search (ILPS) main function.
# ----------------------------------------------------------------------------------

def ILPS(G, k=2, delta=10, nu=3, max_iter=100):
    """
    ILPS repeatedly applies local search and plateau search moves
    to improve the incumbent solution.

    Parameters:
      G       : The graph (NetworkX graph) instance.
      k       : The order for the local search (typically 2 or 3).
      delta   : The penalty delay parameter.
      nu      : The parameter controlling the kick (perturbation).
      max_iter: Maximum number of ILPS iterations.

    Returns:
      S_star: The (hopefully improved) minimal independent dominating set.
    """
    # Construct an initial solution by a greedy heuristic.
    S = greedy_initial_solution(G)
    S = extend_to_maximal(G, S)
    S_star = set(S)
    penalty = {v: 0 for v in G.nodes()}
    penalty = update_penalty(penalty, S, delta)
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1
        # Local search (for example, using 2-swap moves)
        S = local_search(S, G, k)
        # Plateau search move
        S = plateau_search(S, G)
        # Update the best (incumbent) solution if an improvement is found
        if len(S) < len(S_star):
            S_star = set(S)
        # Apply perturbation (kick) for diversification
        S = kick(S_star, penalty, G, nu)
        penalty = update_penalty(penalty, S, delta)
    return S_star

# ----------------------------------------------------------------------------------
# Example usage (test on a small random graph)
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create a random graph (Erdös-Rényi model)
    G = nx.grid_2d_graph(3, 3)
    # Run ILPS with chosen parameters
    S_star = ILPS(G, k=2, delta=10, nu=3, max_iter=100)
    print("Minimum Independent Dominating Set found:")
    print(S_star)
    print("Size:", len(S_star))
    # Verify that the solution is indeed a maximal independent set.
    print("Is valid solution:", is_maximal_independent(G, S_star))

    support = [1 if node in S_star else 0 for node in G.nodes()]
    nx.draw(G, with_labels=True, node_color=support)
    plt.show()
