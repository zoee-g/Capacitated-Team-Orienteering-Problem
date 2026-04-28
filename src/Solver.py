import random

SEED = 42
random.seed(SEED)


def route_cost(model, route):
    return sum(model.cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def route_load(model, route):
    return sum(model.nodes[i].demand for i in route if i != 0)


def route_profit(model, route):
    return sum(model.nodes[i].profit for i in route if i != 0)


def can_insert(model, route, node_id, position):
    new_route = route[:position] + [node_id] + route[position:]
    return (
        route_load(model, new_route) <= model.capacity
        and route_cost(model, new_route) <= model.t_max
    )


def best_insertion(model, routes, node_id):
    best = None

    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            new_route = route[:pos] + [node_id] + route[pos:]
            load = route_load(model, new_route)
            cost = route_cost(model, new_route)

            if load <= model.capacity and cost <= model.t_max:
                old_cost = route_cost(model, route)
                increase = cost - old_cost
                score = model.nodes[node_id].profit / (increase + 1e-6)

                if best is None or score > best[0]:
                    best = (score, r_idx, pos)

    return best


def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")


def solve(model, solution_file, enforce_mandatory=True):
    routes = [[0, 0] for _ in range(model.vehicles)]
    inserted = set()

    if enforce_mandatory:
        mandatory_nodes = [
            node.id for node in model.nodes
            if node.isMandatory and not node.isDepot
        ]

        for node_id in mandatory_nodes:
            best = best_insertion(model, routes, node_id)
            if best is None:
                print(f"WARNING: Mandatory node {node_id} could not be inserted.")
                continue

            _, r_idx, pos = best
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)

    candidates = [
        node.id for node in model.nodes
        if not node.isDepot and node.id not in inserted
    ]

    candidates.sort(
        key=lambda i: model.nodes[i].profit / (model.nodes[i].demand + 1),
        reverse=True
    )

    improved = True
    while improved:
        improved = False
        best_global = None

        for node_id in candidates:
            if node_id in inserted:
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

    routes = [route for route in routes if len(route) > 2]

    write_solution(routes, solution_file)

    print(f"Solution written to {solution_file}")