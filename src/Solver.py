import random

SEEDS = [4, 8, 15, 16, 23, 42]
BIG_NUMBER = 10000
RCL_SIZE = 10


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


def best_insertion(model, routes, node_id):
    best = None

    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            new_route = route[:pos] + [node_id] + route[pos:]

            if is_route_feasible(model, new_route):
                increase = route_cost(model, new_route) - route_cost(model, route)
                score = model.nodes[node_id].profit * BIG_NUMBER - increase

                if best is None or score > best[0]:
                    best = (score, r_idx, pos)

    return best


def greedy_construction(model, enforce_mandatory=True):
    routes = [[0, 0] for _ in range(model.vehicles)]
    inserted = set()

    # Mandatory nodes first
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
            best = best_insertion(model, routes, node_id)

            if best is None:
                print(f"WARNING: Mandatory node {node_id} could not be inserted.")
                continue

            _, r_idx, pos = best
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)

    # Optional nodes with RCL
    while True:
        candidates = [
            node.id for node in model.nodes
            if not node.isDepot and node.id not in inserted
        ]

        scored_candidates = []

        for node_id in candidates:
            best = best_insertion(model, routes, node_id)
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


def improve_by_replacement(model, routes, inserted, enforce_mandatory=True):
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
                                    move_score,
                                    r_idx,
                                    removed_node,
                                    candidate,
                                    new_route
                                )

        if best_move is not None:
            _, r_idx, removed_node, candidate, new_route = best_move

            routes[r_idx] = new_route
            inserted.remove(removed_node)
            inserted.add(candidate)
            improved = True

    return routes, inserted


def improve_by_extra_insertions(model, routes, inserted):
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


def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            if len(route) > 2:
                f.write(" ".join(map(str, route)) + "\n")


def run_single_restart(model, enforce_mandatory, seed):
    random.seed(seed)

    routes, inserted = greedy_construction(model, enforce_mandatory)

    initial_profit = total_profit(model, routes)
    initial_cost = total_cost(model, routes)

    routes = improve_routes_with_2opt(model, routes)

    routes, inserted = improve_by_replacement(
        model,
        routes,
        inserted,
        enforce_mandatory
    )

    routes, inserted = improve_by_extra_insertions(model, routes, inserted)

    routes = improve_routes_with_2opt(model, routes)

    final_profit = total_profit(model, routes)
    final_cost = total_cost(model, routes)

    return routes, initial_profit, initial_cost, final_profit, final_cost


def solve(model, solution_file, enforce_mandatory=True):
    print("Running multi-start greedy + local search...")

    best_routes = None
    best_profit = -1
    best_cost = float("inf")

    for seed in SEEDS:
        print(f"\n--- Restart with seed {seed} ---")

        routes, initial_profit, initial_cost, final_profit, final_cost = run_single_restart(
            model,
            enforce_mandatory,
            seed
        )

        print(f"Initial profit: {initial_profit}, Initial cost: {initial_cost:.2f}")
        print(f"Final profit:   {final_profit}, Final cost:   {final_cost:.2f}")

        valid = validate_internal(model, routes, enforce_mandatory)

        if not valid:
            print("Restart skipped: internal validation failed.")
            continue

        if final_profit > best_profit or (
                final_profit == best_profit and final_cost < best_cost
        ):
            best_routes = [route[:] for route in routes]
            best_profit = final_profit
            best_cost = final_cost

    routes = [route for route in best_routes if len(route) > 2]

    write_solution(routes, solution_file)

    print("\n🏆 Best solution selected")
    print(f"Best profit before validator: {best_profit}")
    print(f"Best cost before validator: {best_cost:.2f}")
    print(f"Solution written to {solution_file}")
