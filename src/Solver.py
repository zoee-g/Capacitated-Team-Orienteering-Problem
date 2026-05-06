"""
Solver.py
---------
Τεχνικές που χρησιμοποιούνται:
  - Multi-start Greedy Construction
  - Restricted Candidate List (RCL)
  - Mandatory-first Insertion
  - Adaptive Memory Pool (Frequency Memory)
  - Memory-biased Construction
  - Local Search (Replacement, Extra Insertion)
  - 2-opt Intra-route Optimization
  - Multi-seed Diversification
  - Internal Feasibility Validation
"""

import random
import copy

SEEDS = [4, 8, 15, 16, 23, 42]
BIG_NUMBER = 10000
RCL_SIZE = 10

# ─────────────────────────────────────────────────────────────────────────────
# Route utility functions
# ─────────────────────────────────────────────────────────────────────────────

def route_cost(model, route):
    return sum(model.cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def route_load(model, route):
    return sum(model.nodes[i].demand for i in route if i != 0)


def route_profit(model, route):
    return sum(model.nodes[i].profit for i in route if i != 0)


def total_profit(model, routes):
    return sum(route_profit(model, route) for route in routes)


def total_cost(model, routes):
    return sum(route_cost(model, route) for route in routes)


def is_route_feasible(model, route):
    return (
        route[0] == 0
        and route[-1] == 0
        and route_load(model, route) <= model.capacity
        and route_cost(model, route) <= model.t_max
    )


# ─────────────────────────────────────────────────────────────────────────────
# Insertion helpers
# ─────────────────────────────────────────────────────────────────────────────

def best_insertion(model, routes, node_id, frequency_memory=None, memory_bias_weight=0.0):
    """
    Find the best feasible insertion of node_id across all routes and positions.

    Scoring formula:
      - Without memory: BIG_NUMBER × profit − route_cost_increase
      - With memory:    BIG_NUMBER × profit − route_cost_increase
                        + memory_bias_weight × freq(node_id) × BIG_NUMBER

    Parameters
    ----------
    frequency_memory    : dict {node_id: visit_count} or None
    memory_bias_weight  : float, how strongly the frequency bonus influences scoring
    """
    best = None
    freq_bonus = 0.0

    if frequency_memory is not None and memory_bias_weight > 0.0:
        freq = frequency_memory.get(node_id, 0)
        freq_bonus = memory_bias_weight * freq * BIG_NUMBER

    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            new_route = route[:pos] + [node_id] + route[pos:]

            if is_route_feasible(model, new_route):
                increase = route_cost(model, new_route) - route_cost(model, route)
                score = model.nodes[node_id].profit * BIG_NUMBER - increase + freq_bonus

                if best is None or score > best[0]:
                    best = (score, r_idx, pos)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

def greedy_construction(model, enforce_mandatory=True,
                        frequency_memory=None, use_memory_bias=False):
    """
    Build a feasible solution using greedy RCL construction.

    - Mandatory nodes are inserted first (sorted by profit/demand ratio).
    - Optional nodes are inserted using a Restricted Candidate List.
    - If use_memory_bias=True and frequency_memory is populated, the
      construction scoring is biased toward nodes that appeared in previous
      good solutions (Adaptive Memory Pool mechanism).
    """
    routes = [[0, 0] for _ in range(model.vehicles)]
    inserted = set()

    memory_bias_weight = 10.0 if use_memory_bias and frequency_memory else 0.0

    # ── Mandatory nodes first ──────────────────────────────────────────────
    if enforce_mandatory:
        mandatory_nodes = [
            node.id for node in model.nodes
            if node.isMandatory and not node.isDepot
        ]
        mandatory_nodes.sort(
            key=lambda i: model.nodes[i].profit / (model.nodes[i].demand + 1),
            reverse=True
        )
        for node_id in mandatory_nodes:
            best = best_insertion(
                model, routes, node_id,
                frequency_memory=frequency_memory,
                memory_bias_weight=memory_bias_weight
            )
            if best is None:
                continue
            _, r_idx, pos = best
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)

    # ── Optional nodes with RCL (memory-biased or plain) ──────────────────
    while True:
        candidates = [
            node.id for node in model.nodes
            if not node.isDepot and node.id not in inserted
        ]

        scored_candidates = []
        for node_id in candidates:
            best = best_insertion(
                model, routes, node_id,
                frequency_memory=frequency_memory,
                memory_bias_weight=memory_bias_weight
            )
            if best is not None:
                score, r_idx, pos = best
                scored_candidates.append((score, node_id, r_idx, pos))

        if not scored_candidates:
            break

        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        rcl = scored_candidates[:RCL_SIZE]
        selected = random.choice(rcl)
        _, node_id, r_idx, pos = selected
        routes[r_idx].insert(pos, node_id)
        inserted.add(node_id)

    return routes, inserted


# ─────────────────────────────────────────────────────────────────────────────
# Local Search operators
# ─────────────────────────────────────────────────────────────────────────────

def improve_by_replacement(model, routes, inserted, enforce_mandatory=True):
    """
    Iteratively replace a routed node with an unvisited node if profit improves.
    Mandatory nodes are never removed when enforce_mandatory=True.
    """
    improved = True
    while improved:
        improved = False
        best_move = None

        unvisited = [
            node.id for node in model.nodes
            if node.id != 0 and node.id not in inserted
        ]

        for r_idx, route in enumerate(routes):
            for remove_pos in range(1, len(route) - 1):
                removed_node = route[remove_pos]

                if enforce_mandatory and model.nodes[removed_node].isMandatory:
                    continue

                route_without = route[:remove_pos] + route[remove_pos + 1:]

                for candidate in unvisited:
                    for insert_pos in range(1, len(route_without)):
                        new_route = (
                            route_without[:insert_pos]
                            + [candidate]
                            + route_without[insert_pos:]
                        )
                        if not is_route_feasible(model, new_route):
                            continue

                        profit_gain = route_profit(model, new_route) - route_profit(model, route)
                        cost_change = route_cost(model, new_route) - route_cost(model, route)
                        move_score = profit_gain * BIG_NUMBER - cost_change

                        if profit_gain > 0:
                            if best_move is None or move_score > best_move[0]:
                                best_move = (
                                    move_score, r_idx,
                                    removed_node, candidate, new_route
                                )

        if best_move is not None:
            _, r_idx, removed_node, candidate, new_route = best_move
            routes[r_idx] = new_route
            inserted.remove(removed_node)
            inserted.add(candidate)
            improved = True

    return routes, inserted


def improve_by_extra_insertions(model, routes, inserted):
    """
    Greedily insert unvisited nodes into routes as long as feasible and beneficial.
    """
    improved = True
    while improved:
        improved = False
        best_global = None

        for node in model.nodes:
            node_id = node.id
            if node_id == 0 or node_id in inserted:
                continue

            best = best_insertion(model, routes, node_id)
            if best is not None:
                score, r_idx, pos = best
                if best_global is None or score > best_global[0]:
                    best_global = (score, node_id, r_idx, pos)

        if best_global is not None:
            _, node_id, r_idx, pos = best_global
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)
            improved = True

    return routes, inserted


def two_opt_route(model, route):
    """Apply 2-opt improvement to a single route."""
    best_route = route[:]
    best_cost = route_cost(model, best_route)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                candidate = (
                    best_route[:i]
                    + list(reversed(best_route[i:j + 1]))
                    + best_route[j + 1:]
                )
                candidate_cost = route_cost(model, candidate)
                if candidate_cost < best_cost and is_route_feasible(model, candidate):
                    best_route = candidate
                    best_cost = candidate_cost
                    improved = True

    return best_route


def improve_routes_with_2opt(model, routes):
    for r_idx in range(len(routes)):
        routes[r_idx] = two_opt_route(model, routes[r_idx])
    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Memory Pool
# ─────────────────────────────────────────────────────────────────────────────

def update_frequency_memory(frequency_memory, routes):
    """
    Increment the visit counter for every non-depot node
    that appears in the current solution.

    This is the "learning" step of the Adaptive Memory Pool:
    nodes that consistently appear in good solutions accumulate
    high counts and receive a construction bonus in future restarts.
    """
    for route in routes:
        for node_id in route:
            if node_id != 0:
                frequency_memory[node_id] = frequency_memory.get(node_id, 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_internal(model, routes, enforce_mandatory=True):
    visited = set()

    if len(routes) > model.vehicles:
        return False

    for route in routes:
        if not is_route_feasible(model, route):
            return False
        for node in route[1:-1]:
            if node in visited:
                return False
            visited.add(node)

    if enforce_mandatory:
        mandatory = {
            node.id for node in model.nodes
            if node.isMandatory and not node.isDepot
        }
        if not mandatory.issubset(visited):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            if len(route) > 2:
                f.write(" ".join(map(str, route)) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Single restart
# ─────────────────────────────────────────────────────────────────────────────

def run_single_restart(model, enforce_mandatory, seed,
                       frequency_memory=None, use_memory_bias=False):
    """
    Execute one full construct → improve cycle.

    Parameters
    ----------
    frequency_memory : dict or None
        Cumulative node-visit counts from previous restarts.
    use_memory_bias  : bool
        If True, construction is biased by frequency_memory
        (Adaptive Memory Pool is active).
    """
    random.seed(seed)

    routes, inserted = greedy_construction(
        model,
        enforce_mandatory,
        frequency_memory=frequency_memory,
        use_memory_bias=use_memory_bias
    )

    initial_profit = total_profit(model, routes)
    initial_cost = total_cost(model, routes)

    routes = improve_routes_with_2opt(model, routes)

    routes, inserted = improve_by_replacement(
        model, routes, inserted, enforce_mandatory
    )

    routes, inserted = improve_by_extra_insertions(model, routes, inserted)

    routes = improve_routes_with_2opt(model, routes)

    final_profit = total_profit(model, routes)
    final_cost = total_cost(model, routes)

    return routes, inserted, initial_profit, initial_cost, final_profit, final_cost


# ─────────────────────────────────────────────────────────────────────────────
# Main solve entry point
# ─────────────────────────────────────────────────────────────────────────────

def solve(model, solution_file, enforce_mandatory=True):
    """
    Multi-start solver with Adaptive Memory Pool.

    Restart schedule
    ----------------
    Restart 0 (seed=4)  : plain RCL construction — no memory yet.
    Restarts 1-5        : memory-biased construction using frequency
                          counts accumulated from all previous restarts.

    After each restart the frequency memory is updated, so later
    restarts benefit from a richer pool of information.
    """
    best_routes = None
    best_profit = -1
    best_cost = float("inf")

    # Shared frequency memory — persists across all restarts
    frequency_memory = {}

    for restart_idx, seed in enumerate(SEEDS):
        # Use memory bias from restart 1 onwards (restart 0 has no memory yet)
        use_memory_bias = (restart_idx > 0) and (len(frequency_memory) > 0)

        routes, inserted, initial_profit, initial_cost, final_profit, final_cost = \
            run_single_restart(
                model,
                enforce_mandatory,
                seed,
                frequency_memory=frequency_memory,
                use_memory_bias=use_memory_bias
            )

        valid = validate_internal(model, routes, enforce_mandatory)

        if not valid:
            update_frequency_memory(frequency_memory, routes)
            continue

        # ── Update Adaptive Memory Pool ────────────────────────────────────
        update_frequency_memory(frequency_memory, routes)

        # ── Track global best ──────────────────────────────────────────────
        if final_profit > best_profit or (
            final_profit == best_profit and final_cost < best_cost
        ):
            best_routes = [route[:] for route in routes]
            best_profit = final_profit
            best_cost = final_cost

    routes = [route for route in best_routes if len(route) > 2]

    write_solution(routes, solution_file)
