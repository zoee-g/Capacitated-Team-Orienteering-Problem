"""
Solver.py
---------
Τεχνικές που χρησιμοποιούνται:
- Weighted Greedy Construction
- Restricted Candidate List (RCL)
- Multi-start Search
- Parameter Tuning
- Mandatory-first Heuristic
- 2-opt Route Improvement
- Extra Insertion Heuristic
- Tabu Search
- Aspiration Criterion
- Mini Large Neighborhood Search (Mini-LNS)
- Destroy & Repair
- Profit-first Objective with Cost Tie-breaker
"""

import random

SEEDS = [4, 8, 15, 16, 23, 42]

PARAMETER_SETS = [
    (1.0, 4.0, 5),
    (1.2, 4.0, 5),
    (1.0, 5.0, 5),
    (1.0, 4.0, 8),
]

BIG_NUMBER = 10000

TABU_TENURE = 10
MAX_TABU_ITERATIONS = 100

LNS_ITERATIONS = 40
LNS_MIN_REMOVE = 3
LNS_MAX_REMOVE = 5
LNS_REPAIR_CANDIDATE_LIMIT = 40


def route_cost(model, route):
    return sum(model.cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def route_load(model, route):
    return sum(model.nodes[i].demand for i in route if i != 0)


def route_profit(model, route):
    return sum(model.nodes[i].profit for i in route if i != 0)


def total_profit(model, routes):
    return sum(route_profit(model, r) for r in routes)


def total_cost(model, routes):
    return sum(route_cost(model, r) for r in routes)


def is_route_feasible(model, route):
    return (
            route[0] == 0
            and route[-1] == 0
            and route_load(model, route) <= model.capacity
            and route_cost(model, route) <= model.t_max
    )


def get_inserted_nodes(routes):
    inserted = set()
    for route in routes:
        for node in route[1:-1]:
            inserted.add(node)
    return inserted


def clone_routes(routes):
    return [r[:] for r in routes]


def better_solution(model, a, b):
    if b is None:
        return True

    pa = total_profit(model, a)
    pb = total_profit(model, b)

    if pa > pb:
        return True

    if pa == pb and total_cost(model, a) < total_cost(model, b):
        return True

    return False


def weighted_greedy_construction(
        model,
        enforce_mandatory=True,
        time_weight=1.0,
        capacity_weight=4.0,
        rcl_size=5
):
    routes = [[0, 0] for _ in range(model.vehicles)]
    current_loads = [0] * model.vehicles
    current_times = [0.0] * model.vehicles

    available_customers = list(range(1, model.num_nodes))

    while available_customers:
        candidates = []

        for customer in available_customers:
            actual_profit = model.nodes[customer].profit

            if enforce_mandatory and model.nodes[customer].isMandatory:
                actual_profit += 10**9

            for v_idx in range(model.vehicles):
                time_left_pct = (model.t_max - current_times[v_idx]) / model.t_max
                cap_left_pct = (model.capacity - current_loads[v_idx]) / model.capacity

                alpha = time_weight / (time_left_pct + 0.01)
                beta = capacity_weight / (cap_left_pct + 0.01)

                for pos in range(1, len(routes[v_idx])):
                    prev_node = routes[v_idx][pos - 1]
                    next_node = routes[v_idx][pos]

                    new_load = current_loads[v_idx] + model.nodes[customer].demand
                    if new_load > model.capacity:
                        continue

                    added_dist = (
                            model.cost_matrix[prev_node][customer]
                            + model.cost_matrix[customer][next_node]
                    )
                    removed_dist = model.cost_matrix[prev_node][next_node]

                    cost_increase = added_dist - removed_dist
                    new_time = current_times[v_idx] + cost_increase

                    if new_time > model.t_max:
                        continue

                    normalized_time_cost = cost_increase / model.t_max
                    normalized_demand = model.nodes[customer].demand / model.capacity

                    penalty = alpha * normalized_time_cost + beta * normalized_demand

                    if penalty <= 0:
                        penalty = 0.000001

                    score = actual_profit / penalty
                    candidates.append((score, customer, v_idx, pos, new_load, new_time))

        if not candidates:
            break

        candidates.sort(reverse=True, key=lambda x: x[0])
        rcl = candidates[:rcl_size]

        _, customer, v_idx, pos, new_load, new_time = random.choice(rcl)

        routes[v_idx].insert(pos, customer)
        current_loads[v_idx] = new_load
        current_times[v_idx] = new_time
        available_customers.remove(customer)

    return routes


def two_opt_route(model, route):
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
    for i in range(len(routes)):
        routes[i] = two_opt_route(model, routes[i])
    return routes


def best_extra_insertion(model, routes, inserted):
    best = None

    for node_id in range(1, model.num_nodes):
        if node_id in inserted:
            continue

        for r_idx, route in enumerate(routes):
            old_cost = route_cost(model, route)

            for pos in range(1, len(route)):
                new_route = route[:pos] + [node_id] + route[pos:]

                if not is_route_feasible(model, new_route):
                    continue

                cost_increase = route_cost(model, new_route) - old_cost
                score = model.nodes[node_id].profit * BIG_NUMBER - cost_increase

                if best is None or score > best[0]:
                    best = (score, node_id, r_idx, pos)

    return best


def improve_by_extra_insertions(model, routes):
    inserted = get_inserted_nodes(routes)

    while True:
        best = best_extra_insertion(model, routes, inserted)

        if best is None:
            break

        _, node_id, r_idx, pos = best
        routes[r_idx].insert(pos, node_id)
        inserted.add(node_id)

    return routes


def tabu_replacement_search(model, routes, enforce_mandatory=True):
    current_routes = clone_routes(routes)
    best_routes = clone_routes(routes)

    best_profit = total_profit(model, best_routes)
    best_cost = total_cost(model, best_routes)

    tabu_until = {}

    for iteration in range(MAX_TABU_ITERATIONS):
        inserted = get_inserted_nodes(current_routes)
        unvisited = [i for i in range(1, model.num_nodes) if i not in inserted]

        best_move = None

        for r_idx, route in enumerate(current_routes):
            for remove_pos in range(1, len(route) - 1):
                removed = route[remove_pos]

                if enforce_mandatory and model.nodes[removed].isMandatory:
                    continue

                route_without = route[:remove_pos] + route[remove_pos + 1:]

                for candidate in unvisited:
                    candidate_is_tabu = tabu_until.get(candidate, -1) > iteration

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

                        if profit_gain <= 0:
                            continue

                        new_total_profit = total_profit(model, current_routes) + profit_gain
                        new_total_cost = total_cost(model, current_routes) + cost_change

                        aspiration = (
                                new_total_profit > best_profit
                                or (
                                        new_total_profit == best_profit
                                        and new_total_cost < best_cost
                                )
                        )

                        if candidate_is_tabu and not aspiration:
                            continue

                        move_score = profit_gain * BIG_NUMBER - cost_change

                        if best_move is None or move_score > best_move[0]:
                            best_move = (
                                move_score,
                                r_idx,
                                removed,
                                new_route,
                                new_total_profit,
                                new_total_cost
                            )

        if best_move is None:
            break

        _, r_idx, removed, new_route, new_total_profit, new_total_cost = best_move

        current_routes[r_idx] = new_route
        tabu_until[removed] = iteration + TABU_TENURE

        if new_total_profit > best_profit or (
                new_total_profit == best_profit and new_total_cost < best_cost
        ):
            best_routes = clone_routes(current_routes)
            best_profit = new_total_profit
            best_cost = new_total_cost

    return best_routes


def collect_removable_nodes(model, routes, enforce_mandatory):
    removable = []

    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route) - 1):
            node_id = route[pos]

            if enforce_mandatory and model.nodes[node_id].isMandatory:
                continue

            profit = model.nodes[node_id].profit
            demand = model.nodes[node_id].demand
            density = profit / (demand + 0.000001)

            prev_node = route[pos - 1]
            next_node = route[pos + 1]

            insertion_cost = (
                    model.cost_matrix[prev_node][node_id]
                    + model.cost_matrix[node_id][next_node]
                    - model.cost_matrix[prev_node][next_node]
            )

            removable.append({
                "node": node_id,
                "density": density,
                "insertion_cost": insertion_cost
            })

    return removable


def destroy_solution(model, routes, enforce_mandatory=True):
    new_routes = clone_routes(routes)
    removable = collect_removable_nodes(model, new_routes, enforce_mandatory)

    if not removable:
        return new_routes

    remove_count = random.randint(LNS_MIN_REMOVE, LNS_MAX_REMOVE)
    remove_count = min(remove_count, len(removable))

    strategy = random.choice(["random", "low_density", "expensive"])

    if strategy == "random":
        selected = random.sample(removable, remove_count)
    elif strategy == "low_density":
        removable.sort(key=lambda x: x["density"])
        selected = removable[:remove_count]
    else:
        removable.sort(key=lambda x: x["insertion_cost"], reverse=True)
        selected = removable[:remove_count]

    nodes_to_remove = {x["node"] for x in selected}

    for r_idx in range(len(new_routes)):
        new_routes[r_idx] = [
            node for node in new_routes[r_idx]
            if node == 0 or node not in nodes_to_remove
        ]

        if len(new_routes[r_idx]) == 1:
            new_routes[r_idx].append(0)

    return new_routes


def weighted_best_repair_insertion(
        model,
        routes,
        candidate_nodes,
        time_weight,
        capacity_weight
):
    best = None

    for node_id in candidate_nodes:
        for r_idx, route in enumerate(routes):
            current_time = route_cost(model, route)
            current_load = route_load(model, route)

            time_left_pct = (model.t_max - current_time) / model.t_max
            cap_left_pct = (model.capacity - current_load) / model.capacity

            alpha = time_weight / (time_left_pct + 0.01)
            beta = capacity_weight / (cap_left_pct + 0.01)

            for pos in range(1, len(route)):
                new_route = route[:pos] + [node_id] + route[pos:]

                if not is_route_feasible(model, new_route):
                    continue

                cost_increase = route_cost(model, new_route) - current_time
                normalized_time_cost = cost_increase / model.t_max
                normalized_demand = model.nodes[node_id].demand / model.capacity

                penalty = alpha * normalized_time_cost + beta * normalized_demand

                if penalty <= 0:
                    penalty = 0.000001

                score = model.nodes[node_id].profit / penalty

                if best is None or score > best[0]:
                    best = (score, node_id, r_idx, pos)

    return best


def repair_solution(model, routes, time_weight, capacity_weight):
    repaired = clone_routes(routes)

    while True:
        inserted = get_inserted_nodes(repaired)
        unvisited = [i for i in range(1, model.num_nodes) if i not in inserted]

        if not unvisited:
            break

        unvisited.sort(
            key=lambda i: (
                model.nodes[i].profit / (model.nodes[i].demand + 0.000001),
                model.nodes[i].profit
            ),
            reverse=True
        )

        candidate_nodes = unvisited[:LNS_REPAIR_CANDIDATE_LIMIT]

        best = weighted_best_repair_insertion(
            model,
            repaired,
            candidate_nodes,
            time_weight,
            capacity_weight
        )

        if best is None:
            break

        _, node_id, r_idx, pos = best
        repaired[r_idx].insert(pos, node_id)

    return repaired


def mini_lns_search(
        model,
        routes,
        enforce_mandatory=True,
        time_weight=1.0,
        capacity_weight=4.0
):
    current_routes = clone_routes(routes)
    best_routes = clone_routes(routes)

    for _ in range(LNS_ITERATIONS):
        partial_routes = destroy_solution(model, current_routes, enforce_mandatory)

        candidate_routes = repair_solution(
            model,
            partial_routes,
            time_weight,
            capacity_weight
        )

        candidate_routes = improve_routes_with_2opt(model, candidate_routes)

        if not validate_internal(model, candidate_routes, enforce_mandatory):
            continue

        if better_solution(model, candidate_routes, current_routes):
            current_routes = clone_routes(candidate_routes)

        if better_solution(model, candidate_routes, best_routes):
            best_routes = clone_routes(candidate_routes)

    return best_routes


def validate_internal(model, routes, enforce_mandatory=True):
    if len(routes) > model.vehicles:
        return False

    visited = set()

    for route in routes:
        if len(route) < 2:
            return False

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


def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            if len(route) > 2:
                f.write(" ".join(map(str, route)) + "\n")


def run_single_restart(
        model,
        enforce_mandatory,
        seed,
        time_weight,
        capacity_weight,
        rcl_size
):
    random.seed(seed)

    routes = weighted_greedy_construction(
        model,
        enforce_mandatory,
        time_weight,
        capacity_weight,
        rcl_size
    )

    initial_profit = total_profit(model, routes)
    initial_cost = total_cost(model, routes)

    best_routes = clone_routes(routes)

    routes = improve_routes_with_2opt(model, routes)
    if better_solution(model, routes, best_routes):
        best_routes = clone_routes(routes)

    routes = improve_by_extra_insertions(model, routes)
    if better_solution(model, routes, best_routes):
        best_routes = clone_routes(routes)

    routes = tabu_replacement_search(model, routes, enforce_mandatory)
    if better_solution(model, routes, best_routes):
        best_routes = clone_routes(routes)

    routes = improve_by_extra_insertions(model, routes)
    routes = improve_routes_with_2opt(model, routes)

    if better_solution(model, routes, best_routes):
        best_routes = clone_routes(routes)

    final_profit = total_profit(model, best_routes)
    final_cost = total_cost(model, best_routes)

    return best_routes, initial_profit, initial_cost, final_profit, final_cost


def solve(model, solution_file, enforce_mandatory=True):
    print("Running fast weighted greedy + parameter multi-start + Tabu...")

    best_routes = None
    best_profit = -1
    best_cost = float("inf")
    best_params = None

    run_counter = 0

    for time_weight, capacity_weight, rcl_size in PARAMETER_SETS:
        for seed in SEEDS:
            run_counter += 1
            print(
                f"\n--- Run {run_counter}: seed={seed}, "
                f"TW={time_weight}, CW={capacity_weight}, RCL={rcl_size} ---"
            )

            routes, initial_profit, initial_cost, final_profit, final_cost = run_single_restart(
                model,
                enforce_mandatory,
                seed,
                time_weight,
                capacity_weight,
                rcl_size
            )

            print(f"Initial profit: {initial_profit}, Initial cost: {initial_cost:.2f}")
            print(f"Final profit:   {final_profit}, Final cost:   {final_cost:.2f}")

            if not validate_internal(model, routes, enforce_mandatory):
                print("Run skipped: internal validation failed.")
                continue

            if final_profit > best_profit or (
                    final_profit == best_profit and final_cost < best_cost
            ):
                best_routes = clone_routes(routes)
                best_profit = final_profit
                best_cost = final_cost
                best_params = (seed, time_weight, capacity_weight, rcl_size)
                print(f"✅ New best found: profit={best_profit}, cost={best_cost:.2f}")

    if best_routes is None:
        raise RuntimeError("No valid solution found.")

    print("\nApplying final Mini-LNS only on best solution...")

    seed, time_weight, capacity_weight, rcl_size = best_params
    random.seed(seed + 999)

    best_after_lns = mini_lns_search(
        model,
        best_routes,
        enforce_mandatory,
        time_weight,
        capacity_weight
    )

    if validate_internal(model, best_after_lns, enforce_mandatory):
        if better_solution(model, best_after_lns, best_routes):
            best_routes = clone_routes(best_after_lns)
            best_profit = total_profit(model, best_routes)
            best_cost = total_cost(model, best_routes)
            print(f"✅ Final LNS improved best: profit={best_profit}, cost={best_cost:.2f}")
        else:
            print("Final LNS did not improve best solution.")
    else:
        print("Final LNS skipped: invalid candidate.")

    final_routes = [r for r in best_routes if len(r) > 2]

    write_solution(final_routes, solution_file)

    print("\n🏆 Best solution selected")
    print(f"Best profit before validator: {best_profit}")
    print(f"Best cost before validator: {best_cost:.2f}")
    print(f"Best params: seed={best_params[0]}, TW={best_params[1]}, CW={best_params[2]}, RCL={best_params[3]}")
    print(f"Solution written to {solution_file}")
