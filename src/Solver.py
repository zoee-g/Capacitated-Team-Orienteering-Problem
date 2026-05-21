import random
import time

"""
Solver με alns και μετά tabu.
Λύση πρώτου προβλήματος: 7670
Δεύτερου: 8573
"""


SEEDS = [4, 8, 15, 16, 23, 42]

PARAMETER_SETS = [
    (1.0, 4.0, 5),
    (1.2, 4.0, 5),
    (1.0, 5.0, 5),
    (0.4, 2.0, 4),
]

BIG_NUMBER = 10000

ALNS_ITERATIONS = 600
ALNS_MIN_REMOVE = 4
ALNS_MAX_REMOVE = 14
ALNS_REPAIR_CANDIDATE_LIMIT = 45
REPAIR_RCL_SIZE = 3

ALNS_REWARD_GLOBAL_BEST = 8
ALNS_REWARD_IMPROVEMENT = 4
ALNS_REWARD_ACCEPTED = 1
ALNS_DECAY = 0.90

TABU_TENURE = 12
MAX_TABU_ITERATIONS = 250

TIME_LIMIT_SECONDS = 285


class Solver:
    def __init__(self, model, solution_file, enforce_mandatory=True):
        self.model = model
        self.solution_file = solution_file
        self.enforce_mandatory = enforce_mandatory

        self.allNodes = model.nodes
        self.distanceMatrix = model.cost_matrix
        self.capacity = model.capacity
        self.t_max = model.t_max
        self.number_of_vehicles = model.vehicles
        self.num_nodes = model.num_nodes

        self.bestSolution = None
        self.bestProfit = -1
        self.bestCost = float("inf")
        self.bestParams = None
        self.startTime = None

    def Solve(self):
        self.startTime = time.time()

        print("Starting Metaheuristic Pipeline: GRASP Tuning -> ALNS -> Tabu Search...")

        self.MultiStartConstruction()

        if self.bestSolution is None:
            raise RuntimeError("No valid solution found.")

        print(f" -> Phase 1 Finished. Best Initial Profit from GRASP: {self.bestProfit}")

        seed, time_weight, capacity_weight, rcl_size = self.bestParams
        random.seed(seed + 555)

        self.bestSolution = self.ExecuteALNSSearch(
            self.bestSolution,
            time_weight,
            capacity_weight
        )

        self.bestSolution = self.TabuSearch(self.bestSolution)

        self.bestSolution = self.ApplyExtraInsertions(self.bestSolution)
        self.bestSolution = self.ApplyTwoOptToAllRoutes(self.bestSolution)

        self.bestProfit = self.CalculateTotalProfit(self.bestSolution)
        self.bestCost = self.CalculateTotalCost(self.bestSolution)

        final_routes = [route for route in self.bestSolution if len(route) > 2]
        self.WriteSolution(final_routes)

        end_time = time.time()

        print(f"\n🏆 Optimization Completed in {end_time - self.startTime:.2f} seconds.")
        print(f"Final Best Profit: {self.bestProfit}")
        print(f"Final Best Cost: {self.bestCost:.2f}")

    def TimeExceeded(self):
        return time.time() - self.startTime >= TIME_LIMIT_SECONDS

    def MultiStartConstruction(self):
        for time_weight, capacity_weight, rcl_size in PARAMETER_SETS:
            for seed in SEEDS:
                if self.TimeExceeded():
                    return

                random.seed(seed)

                routes = self.GenerateInitialSolutionWithRCL(
                    time_weight,
                    capacity_weight,
                    rcl_size
                )

                routes = self.ApplyTwoOptToAllRoutes(routes)
                routes = self.ApplyExtraInsertions(routes)
                routes = self.ApplyTwoOptToAllRoutes(routes)

                if not self.ValidateSolution(routes):
                    continue

                profit = self.CalculateTotalProfit(routes)
                cost = self.CalculateTotalCost(routes)

                if profit > self.bestProfit or (profit == self.bestProfit and cost < self.bestCost):
                    self.bestSolution = self.CloneRoutes(routes)
                    self.bestProfit = profit
                    self.bestCost = cost
                    self.bestParams = (seed, time_weight, capacity_weight, rcl_size)

    def GenerateInitialSolutionWithRCL(self, time_weight, capacity_weight, rcl_size):
        routes = [[0, 0] for _ in range(self.number_of_vehicles)]
        current_loads = [0] * self.number_of_vehicles
        current_times = [0.0] * self.number_of_vehicles

        available_customers = list(range(1, self.num_nodes))

        while available_customers:
            candidates = []

            for customer in available_customers:
                actual_profit = self.allNodes[customer].profit

                if self.enforce_mandatory and self.allNodes[customer].isMandatory:
                    actual_profit += 10 ** 9

                for route_index in range(self.number_of_vehicles):
                    time_left_pct = (self.t_max - current_times[route_index]) / self.t_max if self.t_max > 0 else 0
                    capacity_left_pct = (self.capacity - current_loads[route_index]) / self.capacity if self.capacity > 0 else 0

                    alpha = time_weight / (time_left_pct + 0.01)
                    beta = capacity_weight / (capacity_left_pct + 0.01)

                    for position in range(1, len(routes[route_index])):
                        previous_node = routes[route_index][position - 1]
                        next_node = routes[route_index][position]

                        new_load = current_loads[route_index] + self.allNodes[customer].demand

                        if new_load > self.capacity:
                            continue

                        added_cost = (
                                self.distanceMatrix[previous_node][customer]
                                + self.distanceMatrix[customer][next_node]
                        )

                        removed_cost = self.distanceMatrix[previous_node][next_node]
                        cost_change = added_cost - removed_cost
                        new_time = current_times[route_index] + cost_change

                        if new_time > self.t_max:
                            continue

                        normalized_time_cost = cost_change / self.t_max
                        normalized_demand = self.allNodes[customer].demand / self.capacity

                        penalty = alpha * normalized_time_cost + beta * normalized_demand

                        if penalty <= 0:
                            penalty = 0.000001

                        score = actual_profit / penalty

                        candidates.append(
                            (score, customer, route_index, position, new_load, new_time)
                        )

            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda x: x[0])
            restricted_candidate_list = candidates[:rcl_size]
            selected = random.choice(restricted_candidate_list)

            _, customer, route_index, position, new_load, new_time = selected

            routes[route_index].insert(position, customer)
            current_loads[route_index] = new_load
            current_times[route_index] = new_time
            available_customers.remove(customer)

        return routes

    def ExecuteALNSSearch(self, routes, time_weight, capacity_weight):
        print(f" -> Phase 2: Executing ALNS Metaheuristic ({ALNS_ITERATIONS} iterations)...")

        current_solution = self.CloneRoutes(routes)
        best_solution = self.CloneRoutes(routes)

        destroy_weights = {
            "random": 1.0,
            "low_density": 1.0,
            "expensive": 1.0,
        }

        repair_strategies = [
            (1.0, 4.0),
            (0.5, 2.0),
            (1.5, 5.0),
            (0.2, 1.0),
        ]

        for iteration in range(ALNS_ITERATIONS):
            if self.TimeExceeded():
                break

            destroy_operator = self.SelectDestroyOperator(destroy_weights)

            repair_time_weight, repair_capacity_weight = repair_strategies[
                iteration % len(repair_strategies)
                ]

            use_stochastic_repair = iteration % 2 == 0

            partial_solution = self.DestroySolution(
                current_solution,
                destroy_operator
            )

            candidate_solution = self.AdaptiveRepairSolution(
                partial_solution,
                repair_time_weight,
                repair_capacity_weight,
                use_stochastic_repair
            )

            if not self.ValidateSolution(candidate_solution):
                self.UpdateDestroyWeights(destroy_weights, destroy_operator, 0)
                continue

            if self.CalculateTotalProfit(candidate_solution) >= self.CalculateTotalProfit(current_solution):
                candidate_solution = self.ApplyTwoOptToAllRoutes(candidate_solution)

            reward = 0

            if self.IsBetterSolution(candidate_solution, best_solution):
                best_solution = self.CloneRoutes(candidate_solution)
                current_solution = self.CloneRoutes(candidate_solution)
                reward = ALNS_REWARD_GLOBAL_BEST
                print(f"    [ALNS Improvement] Profit upgraded to: {self.CalculateTotalProfit(best_solution)}")

            elif self.IsBetterSolution(candidate_solution, current_solution):
                current_solution = self.CloneRoutes(candidate_solution)
                reward = ALNS_REWARD_IMPROVEMENT

            elif self.CalculateTotalProfit(candidate_solution) == self.CalculateTotalProfit(current_solution):
                if self.CalculateTotalCost(candidate_solution) < self.CalculateTotalCost(current_solution):
                    current_solution = self.CloneRoutes(candidate_solution)
                    reward = ALNS_REWARD_ACCEPTED

            self.UpdateDestroyWeights(destroy_weights, destroy_operator, reward)

        return best_solution

    def SelectDestroyOperator(self, destroy_weights):
        total_weight = sum(destroy_weights.values())
        pick = random.uniform(0, total_weight)

        cumulative = 0.0

        for operator_name, weight in destroy_weights.items():
            cumulative += weight

            if pick <= cumulative:
                return operator_name

        return list(destroy_weights.keys())[-1]

    def UpdateDestroyWeights(self, destroy_weights, selected_operator, reward):
        for operator_name in destroy_weights:
            destroy_weights[operator_name] *= ALNS_DECAY

        destroy_weights[selected_operator] += reward

    def DestroySolution(self, routes, destroy_operator):
        new_routes = self.CloneRoutes(routes)
        removable_nodes = self.CollectRemovableNodes(new_routes)

        if not removable_nodes:
            return new_routes

        remove_count = random.randint(ALNS_MIN_REMOVE, ALNS_MAX_REMOVE)
        remove_count = min(remove_count, len(removable_nodes))

        if destroy_operator == "random":
            selected_nodes = random.sample(removable_nodes, remove_count)

        elif destroy_operator == "low_density":
            removable_nodes.sort(key=lambda x: x["density"])
            selected_nodes = removable_nodes[:remove_count]

        else:
            removable_nodes.sort(key=lambda x: x["insertion_cost"], reverse=True)
            selected_nodes = removable_nodes[:remove_count]

        nodes_to_remove = {item["node"] for item in selected_nodes}

        for route_index in range(len(new_routes)):
            new_routes[route_index] = [
                node for node in new_routes[route_index]
                if node == 0 or node not in nodes_to_remove
            ]

            if len(new_routes[route_index]) == 1:
                new_routes[route_index].append(0)

        return new_routes

    def AdaptiveRepairSolution(self, routes, time_weight, capacity_weight, use_stochastic_repair=True):
        repaired_solution = self.CloneRoutes(routes)

        inserted_nodes = self.GetInsertedNodes(repaired_solution)
        unvisited_nodes = [
            node_id for node_id in range(1, self.num_nodes)
            if node_id not in inserted_nodes
        ]

        mandatory_unvisited = [
            node_id for node_id in unvisited_nodes
            if self.enforce_mandatory and self.allNodes[node_id].isMandatory
        ]

        optional_unvisited = [
            node_id for node_id in unvisited_nodes
            if not (self.enforce_mandatory and self.allNodes[node_id].isMandatory)
        ]

        for node_id in mandatory_unvisited:
            insertion_positions = []

            for route_index, route in enumerate(repaired_solution):
                old_cost = self.CalculateRouteCost(route)

                for position in range(1, len(route)):
                    new_route = route[:position] + [node_id] + route[position:]

                    if self.IsRouteFeasible(new_route):
                        cost_change = self.CalculateRouteCost(new_route) - old_cost
                        insertion_positions.append((cost_change, route_index, position))

            if insertion_positions:
                insertion_positions.sort(key=lambda x: x[0])
                _, route_index, position = insertion_positions[0]
                repaired_solution[route_index].insert(position, node_id)

        optional_unvisited.sort(
            key=lambda node_id: self.allNodes[node_id].profit / (self.allNodes[node_id].demand + 0.000001),
            reverse=True
        )

        while True:
            inserted_nodes = self.GetInsertedNodes(repaired_solution)

            still_unvisited = [
                node_id for node_id in optional_unvisited
                if node_id not in inserted_nodes
            ]

            if not still_unvisited:
                break

            candidate_nodes = still_unvisited[:ALNS_REPAIR_CANDIDATE_LIMIT]
            valid_insertions = []

            for node_id in candidate_nodes:
                for route_index, route in enumerate(repaired_solution):
                    current_time = self.CalculateRouteCost(route)
                    current_load = self.CalculateRouteLoad(route)

                    time_left_pct = (self.t_max - current_time) / self.t_max if self.t_max > 0 else 0
                    capacity_left_pct = (self.capacity - current_load) / self.capacity if self.capacity > 0 else 0

                    alpha = time_weight / (time_left_pct + 0.01)
                    beta = capacity_weight / (capacity_left_pct + 0.01)

                    for position in range(1, len(route)):
                        new_route = route[:position] + [node_id] + route[position:]

                        if not self.IsRouteFeasible(new_route):
                            continue

                        cost_change = self.CalculateRouteCost(new_route) - current_time
                        normalized_time_cost = cost_change / self.t_max
                        normalized_demand = self.allNodes[node_id].demand / self.capacity

                        penalty = alpha * normalized_time_cost + beta * normalized_demand

                        if penalty <= 0:
                            penalty = 0.000001

                        score = self.allNodes[node_id].profit / penalty

                        valid_insertions.append(
                            (score, node_id, route_index, position)
                        )

            if not valid_insertions:
                break

            valid_insertions.sort(reverse=True, key=lambda x: x[0])

            if use_stochastic_repair:
                repair_rcl = valid_insertions[:REPAIR_RCL_SIZE]
                _, node_id, route_index, position = random.choice(repair_rcl)
            else:
                _, node_id, route_index, position = valid_insertions[0]

            repaired_solution[route_index].insert(position, node_id)

        return repaired_solution

    def TabuSearch(self, routes):
        print(f" -> Phase 3: Executing Final Tabu Search intensification ({MAX_TABU_ITERATIONS} iterations)...")

        current_solution = self.CloneRoutes(routes)
        best_solution = self.CloneRoutes(routes)

        best_profit = self.CalculateTotalProfit(best_solution)
        best_cost = self.CalculateTotalCost(best_solution)

        tabu_until = {}

        for iteration in range(MAX_TABU_ITERATIONS):
            if self.TimeExceeded():
                break

            inserted_nodes = self.GetInsertedNodes(current_solution)
            unvisited_nodes = [
                node_id for node_id in range(1, self.num_nodes)
                if node_id not in inserted_nodes
            ]

            best_move = None

            for route_index, route in enumerate(current_solution):
                for remove_position in range(1, len(route) - 1):
                    removed_node = route[remove_position]

                    if self.enforce_mandatory and self.allNodes[removed_node].isMandatory:
                        continue

                    route_without_node = route[:remove_position] + route[remove_position + 1:]

                    for candidate_node in unvisited_nodes:
                        candidate_is_tabu = tabu_until.get(candidate_node, -1) > iteration

                        for insert_position in range(1, len(route_without_node)):
                            new_route = (
                                    route_without_node[:insert_position]
                                    + [candidate_node]
                                    + route_without_node[insert_position:]
                            )

                            if not self.IsRouteFeasible(new_route):
                                continue

                            profit_change = self.CalculateRouteProfit(new_route) - self.CalculateRouteProfit(route)
                            cost_change = self.CalculateRouteCost(new_route) - self.CalculateRouteCost(route)

                            if profit_change <= 0:
                                continue

                            candidate_profit = self.CalculateTotalProfit(current_solution) + profit_change
                            candidate_cost = self.CalculateTotalCost(current_solution) + cost_change

                            aspiration = (
                                    candidate_profit > best_profit
                                    or (
                                            candidate_profit == best_profit
                                            and candidate_cost < best_cost
                                    )
                            )

                            if candidate_is_tabu and not aspiration:
                                continue

                            move_cost = profit_change * BIG_NUMBER - cost_change

                            if best_move is None or move_cost > best_move[0]:
                                best_move = (
                                    move_cost,
                                    route_index,
                                    removed_node,
                                    candidate_node,
                                    new_route,
                                    candidate_profit,
                                    candidate_cost
                                )

            if best_move is None:
                break

            _, route_index, removed_node, added_node, new_route, candidate_profit, candidate_cost = best_move

            current_solution[route_index] = new_route
            tabu_until[removed_node] = iteration + TABU_TENURE

            if candidate_profit > best_profit or (
                    candidate_profit == best_profit and candidate_cost < best_cost
            ):
                best_solution = self.CloneRoutes(current_solution)
                best_profit = candidate_profit
                best_cost = candidate_cost
                print(f"    [Tabu Breakthrough] Profit upgraded to: {best_profit}")

        return best_solution

    def CollectRemovableNodes(self, routes):
        removable_nodes = []

        for route_index, route in enumerate(routes):
            for position in range(1, len(route) - 1):
                node_id = route[position]

                if self.enforce_mandatory and self.allNodes[node_id].isMandatory:
                    continue

                profit = self.allNodes[node_id].profit
                demand = self.allNodes[node_id].demand
                density = profit / (demand + 0.000001)

                previous_node = route[position - 1]
                next_node = route[position + 1]

                insertion_cost = (
                        self.distanceMatrix[previous_node][node_id]
                        + self.distanceMatrix[node_id][next_node]
                        - self.distanceMatrix[previous_node][next_node]
                )

                removable_nodes.append({
                    "node": node_id,
                    "density": density,
                    "insertion_cost": insertion_cost
                })

        return removable_nodes

    def ApplyExtraInsertions(self, routes):
        routes = self.CloneRoutes(routes)
        inserted_nodes = self.GetInsertedNodes(routes)

        while True:
            best_insertion = None

            for node_id in range(1, self.num_nodes):
                if node_id in inserted_nodes:
                    continue

                for route_index, route in enumerate(routes):
                    old_cost = self.CalculateRouteCost(route)

                    for position in range(1, len(route)):
                        new_route = route[:position] + [node_id] + route[position:]

                        if not self.IsRouteFeasible(new_route):
                            continue

                        cost_change = self.CalculateRouteCost(new_route) - old_cost
                        score = self.allNodes[node_id].profit * BIG_NUMBER - cost_change

                        if best_insertion is None or score > best_insertion[0]:
                            best_insertion = (score, node_id, route_index, position)

            if best_insertion is None:
                break

            _, node_id, route_index, position = best_insertion

            routes[route_index].insert(position, node_id)
            inserted_nodes.add(node_id)

        return routes

    def ApplyTwoOptToAllRoutes(self, routes):
        routes = self.CloneRoutes(routes)

        for route_index in range(len(routes)):
            routes[route_index] = self.TwoOptRoute(routes[route_index])

        return routes

    def TwoOptRoute(self, route):
        best_route = route[:]
        best_cost = self.CalculateRouteCost(best_route)

        improved = True

        while improved:
            improved = False

            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    candidate_route = (
                            best_route[:i]
                            + list(reversed(best_route[i:j + 1]))
                            + best_route[j + 1:]
                    )

                    candidate_cost = self.CalculateRouteCost(candidate_route)

                    if candidate_cost < best_cost and self.IsRouteFeasible(candidate_route):
                        best_route = candidate_route
                        best_cost = candidate_cost
                        improved = True

        return best_route

    def IsRouteFeasible(self, route):
        return (
                len(route) >= 2
                and route[0] == 0
                and route[-1] == 0
                and self.CalculateRouteLoad(route) <= self.capacity
                and self.CalculateRouteCost(route) <= self.t_max
        )

    def ValidateSolution(self, routes):
        if len(routes) > self.number_of_vehicles:
            return False

        visited_nodes = set()

        for route in routes:
            if not self.IsRouteFeasible(route):
                return False

            for node_id in route[1:-1]:
                if node_id in visited_nodes:
                    return False

                visited_nodes.add(node_id)

        if self.enforce_mandatory:
            mandatory_nodes = {
                node.id for node in self.allNodes
                if node.isMandatory and not node.isDepot
            }

            if not mandatory_nodes.issubset(visited_nodes):
                return False

        return True

    def IsBetterSolution(self, candidate_solution, current_solution):
        return self.CalculateTotalProfit(candidate_solution) > self.CalculateTotalProfit(current_solution) or (
                self.CalculateTotalProfit(candidate_solution) == self.CalculateTotalProfit(current_solution)
                and self.CalculateTotalCost(candidate_solution) < self.CalculateTotalCost(current_solution)
        )

    def CalculateRouteCost(self, route):
        return sum(self.distanceMatrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    def CalculateRouteLoad(self, route):
        return sum(self.allNodes[node_id].demand for node_id in route if node_id != 0)

    def CalculateRouteProfit(self, route):
        return sum(self.allNodes[node_id].profit for node_id in route if node_id != 0)

    def CalculateTotalCost(self, routes):
        return sum(self.CalculateRouteCost(route) for route in routes)

    def CalculateTotalProfit(self, routes):
        return sum(self.CalculateRouteProfit(route) for route in routes)

    def GetInsertedNodes(self, routes):
        inserted_nodes = set()

        for route in routes:
            for node_id in route[1:-1]:
                inserted_nodes.add(node_id)

        return inserted_nodes

    def CloneRoutes(self, routes):
        return [route[:] for route in routes]

    def WriteSolution(self, routes):
        with open(self.solution_file, "w") as file:
            for route in routes:
                if len(route) > 2:
                    file.write(" ".join(map(str, route)) + "\n")


def solve(model, solution_file, enforce_mandatory=True):
    solver = Solver(model, solution_file, enforce_mandatory)
    solver.Solve()
