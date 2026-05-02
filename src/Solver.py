"""
Solver.py — Improved CTOP Solver
=================================
Combines a greedy construction heuristic with a Tabu Search metaheuristic
adapted for the Capacitated Team Orienteering Problem (CTOP).

Phase 1: Greedy Construction
    Insert mandatory nodes first (if enforced), then greedily insert
    optional nodes sorted by profit/demand ratio, using best-insertion.

Phase 2: Tabu Search Improvement
    Iteratively apply neighbourhood operators to maximize total profit:
      - Relocate: move a customer to a better position
      - Swap: exchange two customers (same or different routes)
      - 2-opt: reverse a segment within a route
      - Insert: add an unvisited profitable customer
      - Remove-Insert (Replace): drop a low-value customer and insert a
        higher-value unvisited customer in its place

    Tabu list prevents cycling. Aspiration criterion overrides tabu if
    a move produces a new global best.

Objective: Maximize total profit collected.
Tiebreaker: Minimize total route time.

Seeds allowed: 4, 8, 15, 16, 23, 42
"""


import random
import time
import copy

seed = 42
random.seed(seed)

# ─── Utility functions ──────────────────────────────────────────────────────

def route_cost(model, route):
    """Total travel time of a route (list of node IDs)."""
    return sum(model.cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def route_load(model, route):
    """Total demand of customers on a route (excluding depot)."""
    return sum(model.nodes[i].demand for i in route if i != 0)


def route_profit(model, route):
    """Total profit of customers on a route (excluding depot)."""
    return sum(model.nodes[i].profit for i in route if i != 0)


def total_profit(model, routes):
    """Total profit across all routes."""
    return sum(route_profit(model, r) for r in routes)


def total_cost(model, routes):
    """Total travel time across all routes."""
    return sum(route_cost(model, r) for r in routes)


def is_route_feasible(model, route):
    """Check capacity and time constraints for a single route."""
    return (route_load(model, route) <= model.capacity and
            route_cost(model, route) <= model.t_max)

def can_insert(model, route, node_id, position):
    """Check if inserting node_id at position in route is feasible."""
    new_route = route[:position] + [node_id] + route[position:]
    return (
        route_load(model, new_route) <= model.capacity
        and route_cost(model, new_route) <= model.t_max
    )


def insertion_cost_delta(model, route, node_id, position):
    """Cost increase from inserting node_id at position."""
    a = route[position - 1]
    b = route[position]
    return (model.cost_matrix[a][node_id] +
            model.cost_matrix[node_id][b] -
            model.cost_matrix[a][b])


def get_all_served(routes):
    """Return set of all served customer IDs (excluding depot 0)."""
    served = set()
    for r in routes:
        for n in r:
            if n != 0:
                served.add(n)
    return served


def write_solution(routes, solution_file):
    """Write solution to file in required format."""
    with open(solution_file, "w") as f:
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")


# ─── Phase 1: Greedy Construction ───────────────────────────────────────────

def best_insertion(model, routes, node_id):
    """
    Find the best (route_index, position) to insert node_id.
    Score = profit / (cost_increase + epsilon).
    Returns (score, route_idx, position) or None if infeasible.
    """
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


def cheapest_insertion(model, routes, node_id):
    """
    Find the position with minimum cost increase (for mandatory nodes).
    Returns (cost_increase, route_idx, position) or None.
    """
    best = None
    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            if not can_insert(model, route, node_id, pos):
                continue
            delta = insertion_cost_delta(model, route, node_id, pos)
            if best is None or delta < best[0]:
                best = (delta, r_idx, pos)
    return best


def greedy_construct(model, enforce_mandatory=True):
    """
    Build an initial feasible solution using greedy insertion.

    1. If enforce_mandatory: insert mandatory nodes first (cheapest insertion)
    2. Then greedily insert optional nodes sorted by profit/demand ratio
       using best profit-per-cost-increase scoring.
    """
    routes = [[0, 0] for _ in range(model.vehicles)]
    inserted = set()

    # --- Insert mandatory nodes first ---
    if enforce_mandatory:
        mandatory = [n.id for n in model.nodes if n.isMandatory and not n.isDepot]
        # Sort mandatory by demand (ascending) to pack them more easily
        mandatory.sort(key=lambda i: model.nodes[i].demand)

        for node_id in mandatory:
            result = cheapest_insertion(model, routes, node_id)
            if result is None:
                print(f"WARNING: Mandatory node {node_id} could not be inserted!")
                continue
            _, r_idx, pos = result
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)

    # --- Greedily insert optional nodes ---
    candidates = [
        n.id for n in model.nodes
        if not n.isDepot and n.id not in inserted
    ]
    candidates.sort(
        key=lambda i: model.nodes[i].profit / (model.nodes[i].demand + 1),
        reverse=True,
    )

    improved = True
    while improved:
        improved = False
        best_global = None
        for node_id in candidates:
            if node_id in inserted:
                continue
            result = best_insertion(model, routes, node_id)
            if result is not None:
                score, r_idx, pos = result
                if best_global is None or score > best_global[0]:
                    best_global = (score, node_id, r_idx, pos)
        if best_global is not None:
            _, node_id, r_idx, pos = best_global
            routes[r_idx].insert(pos, node_id)
            inserted.add(node_id)
            improved = True

    # Remove empty routes
    routes = [r for r in routes if len(r) > 2]
    return routes


# ─── Phase 2: Tabu Search Improvement ───────────────────────────────────────

class TabuSearchCTOP:
    """
    Tabu Search metaheuristic adapted for CTOP (profit maximization).

    Operators:
      1. Relocate — move customer to better position (same/different route)
      2. Swap — exchange two customers
      3. 2-opt — reverse segment within a route
      4. Insert — add an unvisited customer
      5. Replace — remove a served customer and insert a better unvisited one

    The objective is to MAXIMIZE profit, with total time as tiebreaker.
    """

    def __init__(self, model, routes, enforce_mandatory=True):
        self.model = model
        self.routes = [r[:] for r in routes]  # deep copy
        self.enforce_mandatory = enforce_mandatory

        # Identify mandatory set
        self.mandatory_set = set()
        if enforce_mandatory:
            self.mandatory_set = {
                n.id for n in model.nodes if n.isMandatory and not n.isDepot
            }

        # Tabu list: node_id -> iteration until which it's tabu
        self.tabu = {}
        self.min_tenure = 5
        self.max_tenure = 15

        # Best solution tracking
        self.best_routes = None
        self.best_profit = -1
        self.best_time = float('inf')

        self._update_best()

    def _served(self):
        return get_all_served(self.routes)

    def _unserved(self):
        all_customers = {n.id for n in self.model.nodes if not n.isDepot}
        return all_customers - self._served()

    def _current_profit(self):
        return total_profit(self.model, self.routes)

    def _current_cost(self):
        return total_cost(self.model, self.routes)

    def _update_best(self):
        p = self._current_profit()
        c = self._current_cost()
        if (p > self.best_profit) or (p == self.best_profit and c < self.best_time):
            self.best_profit = p
            self.best_time = c
            self.best_routes = [r[:] for r in self.routes]

    def _is_tabu(self, node_id, iteration):
        return self.tabu.get(node_id, -1) > iteration

    def _set_tabu(self, node_id, iteration):
        self.tabu[node_id] = iteration + random.randint(self.min_tenure, self.max_tenure)

    def _is_protected(self, node_id):
        """Mandatory nodes cannot be removed."""
        return node_id in self.mandatory_set

    # --- Operator 1: Relocate ---
    def _try_relocate(self, iteration):
        """Try all relocations, return the best improving or least-worsening move."""
        best_move = None  # (profit_delta, cost_delta, action_func)
        m = self.model

        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                node_id = route[ni]
                if self._is_tabu(node_id, iteration):
                    continue

                # Cost of removing node from current position
                a, c = route[ni - 1], route[ni + 1]
                remove_delta = (m.cost_matrix[a][c]
                                - m.cost_matrix[a][node_id]
                                - m.cost_matrix[node_id][c])

                # Try inserting into every other position
                for rj, target in enumerate(self.routes):
                    for pj in range(1, len(target)):
                        if ri == rj and pj in (ni, ni + 1):
                            continue

                        # Build test route for target
                        if ri == rj:
                            test = route[:ni] + route[ni + 1:]
                            # Adjust position
                            adj_pos = pj if ni > pj else pj - 1
                            test = test[:adj_pos] + [node_id] + test[adj_pos:]
                        else:
                            test = target[:pj] + [node_id] + target[pj:]

                        # Check feasibility
                        if route_load(m, test) > m.capacity:
                            continue
                        if route_cost(m, test) > m.t_max:
                            continue
                        if ri == rj:
                            # Check the modified route
                            pass  # test is already the full modified route
                        else:
                            # Check origin route without the node
                            origin_test = route[:ni] + route[ni + 1:]
                            # origin always feasible (removing a node)

                        # Profit doesn't change for relocate, but cost might improve
                        if ri == rj:
                            new_cost = route_cost(m, test)
                            old_cost = route_cost(m, route)
                            cost_delta = new_cost - old_cost
                        else:
                            old_origin_cost = route_cost(m, route)
                            new_origin_cost = route_cost(m, route[:ni] + route[ni + 1:])
                            old_target_cost = route_cost(m, target)
                            new_target_cost = route_cost(m, test)
                            cost_delta = (new_origin_cost + new_target_cost) - (old_origin_cost + old_target_cost)

                        profit_delta = 0  # relocate doesn't change profit

                        if best_move is None or cost_delta < best_move[1]:
                            best_move = (profit_delta, cost_delta, ('relocate', ri, ni, rj, pj, node_id))

        return best_move

    # --- Operator 2: Swap ---
    def _try_swap(self, iteration):
        """Try swapping two customers between routes."""
        best_move = None
        m = self.model

        for ri in range(len(self.routes)):
            r1 = self.routes[ri]
            for rj in range(ri, len(self.routes)):
                r2 = self.routes[rj]
                for ni in range(1, len(r1) - 1):
                    start_nj = ni + 1 if ri == rj else 1
                    for nj in range(start_nj, len(r2) - 1):
                        n1 = r1[ni]
                        n2 = r2[nj]

                        if self._is_tabu(n1, iteration) or self._is_tabu(n2, iteration):
                            continue

                        # Build test routes
                        if ri == rj:
                            test = r1[:]
                            test[ni] = n2
                            test[nj] = n1
                            if route_load(m, test) > m.capacity:
                                continue
                            if route_cost(m, test) > m.t_max:
                                continue
                            cost_delta = route_cost(m, test) - route_cost(m, r1)
                        else:
                            test1 = r1[:]
                            test1[ni] = n2
                            test2 = r2[:]
                            test2[nj] = n1
                            if route_load(m, test1) > m.capacity:
                                continue
                            if route_load(m, test2) > m.capacity:
                                continue
                            if route_cost(m, test1) > m.t_max:
                                continue
                            if route_cost(m, test2) > m.t_max:
                                continue
                            cost_delta = ((route_cost(m, test1) + route_cost(m, test2))
                                          - (route_cost(m, r1) + route_cost(m, r2)))

                        # Profit doesn't change in swap
                        if best_move is None or cost_delta < best_move[1]:
                            best_move = (0, cost_delta, ('swap', ri, ni, rj, nj))

        return best_move

    # --- Operator 3: 2-opt (intra-route) ---
    def _try_two_opt(self, iteration):
        """Try reversing a segment within a route to reduce cost."""
        best_move = None
        m = self.model

        for ri, route in enumerate(self.routes):
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Reverse segment route[i..j]
                    test = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_cost = route_cost(m, test)
                    old_cost = route_cost(m, route)
                    if new_cost > m.t_max:
                        continue
                    cost_delta = new_cost - old_cost
                    if best_move is None or cost_delta < best_move[1]:
                        best_move = (0, cost_delta, ('2opt', ri, i, j))

        return best_move

    # --- Operator 4: Insert unvisited customer ---
    def _try_insert(self, iteration):
        """Try inserting an unvisited customer into any route."""
        best_move = None
        m = self.model
        unserved = self._unserved()

        for node_id in unserved:
            if self._is_tabu(node_id, iteration):
                continue
            profit = m.nodes[node_id].profit
            for ri, route in enumerate(self.routes):
                if route_load(m, route) + m.nodes[node_id].demand > m.capacity:
                    continue
                for pos in range(1, len(route)):
                    if not can_insert(m, route, node_id, pos):
                        continue
                    delta = insertion_cost_delta(m, route, node_id, pos)
                    # Score: profit gained (primary), cost increase (secondary)
                    if best_move is None or profit > best_move[0] or (profit == best_move[0] and delta < best_move[1]):
                        best_move = (profit, delta, ('insert', ri, pos, node_id))

        return best_move

    # --- Operator 5: Replace (remove + insert) ---
    def _try_replace(self, iteration):
        """
        Remove a served customer and insert a more profitable unvisited one.
        Only considers replacements that increase total profit.
        """
        best_move = None
        m = self.model
        unserved = self._unserved()

        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                old_id = route[ni]
                if self._is_protected(old_id):
                    continue
                if self._is_tabu(old_id, iteration):
                    continue

                old_profit = m.nodes[old_id].profit
                # Route without old_id
                reduced = route[:ni] + route[ni + 1:]

                for new_id in unserved:
                    new_profit = m.nodes[new_id].profit
                    profit_delta = new_profit - old_profit
                    if profit_delta <= 0:
                        continue
                    if self._is_tabu(new_id, iteration):
                        continue

                    # Try inserting new_id into the reduced route
                    for pos in range(1, len(reduced)):
                        test = reduced[:pos] + [new_id] + reduced[pos:]
                        if route_load(m, test) > m.capacity:
                            continue
                        if route_cost(m, test) > m.t_max:
                            continue
                        cost_delta = route_cost(m, test) - route_cost(m, route)
                        if best_move is None or profit_delta > best_move[0] or (
                                profit_delta == best_move[0] and cost_delta < best_move[1]):
                            best_move = (profit_delta, cost_delta, ('replace', ri, ni, old_id, pos, new_id))

        return best_move

    # --- Apply moves ---
    def _apply_move(self, move_info, iteration):
        """Apply a move and update tabu list."""
        kind = move_info[0]

        if kind == 'relocate':
            _, ri, ni, rj, pj, node_id = move_info
            if ri == rj:
                route = self.routes[ri]
                route.pop(ni)
                adj = pj if ni > pj else pj - 1
                route.insert(adj, node_id)
            else:
                node_id = self.routes[ri].pop(ni)
                self.routes[rj].insert(pj, node_id)
            self._set_tabu(node_id, iteration)

        elif kind == 'swap':
            _, ri, ni, rj, nj = move_info
            if ri == rj:
                self.routes[ri][ni], self.routes[ri][nj] = self.routes[ri][nj], self.routes[ri][ni]
            else:
                self.routes[ri][ni], self.routes[rj][nj] = self.routes[rj][nj], self.routes[ri][ni]
            self._set_tabu(self.routes[ri][ni], iteration)
            self._set_tabu(self.routes[rj][nj], iteration)

        elif kind == '2opt':
            _, ri, i, j = move_info
            self.routes[ri] = self.routes[ri][:i] + self.routes[ri][i:j + 1][::-1] + self.routes[ri][j + 1:]

        elif kind == 'insert':
            _, ri, pos, node_id = move_info
            self.routes[ri].insert(pos, node_id)
            self._set_tabu(node_id, iteration)

        elif kind == 'replace':
            _, ri, ni, old_id, pos, new_id = move_info
            self.routes[ri].pop(ni)
            # Adjust position after removal
            adj_pos = pos if ni >= pos else pos - 1
            self.routes[ri].insert(adj_pos, new_id)
            self._set_tabu(old_id, iteration)
            self._set_tabu(new_id, iteration)

        # Clean up empty routes
        self.routes = [r for r in self.routes if len(r) > 2]
        # Ensure we still have routes for potential future inserts
        if len(self.routes) < self.model.vehicles:
            while len(self.routes) < self.model.vehicles:
                self.routes.append([0, 0])

    def run(self, max_iterations=3000, time_limit=270):
        """
        Main Tabu Search loop.

        At each iteration, evaluate all operators and apply the best move that:
          1. Increases profit (highest priority)
          2. Or reduces total time at equal profit
          3. Is not tabu (unless aspiration criterion met)
        """
        start_time = time.time()
        no_improve = 0
        max_no_improve = 300

        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                print(f"  Time limit reached at iteration {iteration}")
                break

            # Collect candidate moves from all operators
            candidates = []

            # Profit-improving operators (high priority)
            insert_move = self._try_insert(iteration)
            if insert_move is not None:
                candidates.append(insert_move)

            replace_move = self._try_replace(iteration)
            if replace_move is not None:
                candidates.append(replace_move)

            # Cost-improving operators (lower priority, same profit)
            relocate_move = self._try_relocate(iteration)
            if relocate_move is not None:
                candidates.append(relocate_move)

            swap_move = self._try_swap(iteration)
            if swap_move is not None:
                candidates.append(swap_move)

            two_opt_move = self._try_two_opt(iteration)
            if two_opt_move is not None:
                candidates.append(two_opt_move)

            if not candidates:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue

            # Select best: maximize profit_delta, then minimize cost_delta
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best = candidates[0]

            # Only apply if it improves profit or reduces cost at same profit
            if best[0] > 0 or best[1] < -1e-6:
                self._apply_move(best[2], iteration)
                self._update_best()
                no_improve = 0
                if iteration % 100 == 0:
                    print(f"  Iter {iteration}: profit={self._current_profit()}, "
                          f"cost={self._current_cost():.1f}, best_profit={self.best_profit}")
            else:
                no_improve += 1
                # Apply a non-improving move anyway (tabu search accepts worsening)
                # but only cost-worsening moves, not profit-worsening
                if best[0] == 0 and no_improve < max_no_improve:
                    self._apply_move(best[2], iteration)

                if no_improve > max_no_improve:
                    break

        # Perturbation phase: try random restarts with different seeds
        self._perturbation_phase(start_time, time_limit)

        self.routes = [r for r in self.best_routes if len(r) > 2]
        return self.routes

    def _perturbation_phase(self, start_time, time_limit):
        """
        ILS-style perturbation: perturb the best solution and re-optimize.
        """
        perturbation_count = 0

        while time.time() - start_time < time_limit - 10:
            perturbation_count += 1

            # Start from best solution
            self.routes = [r[:] for r in self.best_routes]

            # Perturb: randomly remove some nodes and re-insert
            served = [n for n in self._served() if not self._is_protected(n)]
            if not served:
                break

            num_remove = max(3, len(served) // 5)
            to_remove = random.sample(served, min(num_remove, len(served)))

            for node_id in to_remove:
                for ri, route in enumerate(self.routes):
                    if node_id in route:
                        route.remove(node_id)
                        break

            # Clean empty routes
            self.routes = [r for r in self.routes if len(r) > 2]
            while len(self.routes) < self.model.vehicles:
                self.routes.append([0, 0])

            # Re-insert removed nodes + try unserved nodes
            all_candidates = list(to_remove) + list(self._unserved())
            random.shuffle(all_candidates)

            # Sort by profit descending
            all_candidates.sort(key=lambda i: self.model.nodes[i].profit, reverse=True)

            for node_id in all_candidates:
                if node_id in self._served():
                    continue
                result = best_insertion(self.model, self.routes, node_id)
                if result is not None:
                    _, r_idx, pos = result
                    self.routes[r_idx].insert(pos, node_id)

            # Clear tabu list for fresh local search
            self.tabu.clear()

            # Quick local search
            for iteration in range(500):
                if time.time() - start_time > time_limit - 5:
                    break

                improved = False

                # Try 2-opt
                two_opt_move = self._try_two_opt(iteration + 10000)
                if two_opt_move is not None and two_opt_move[1] < -1e-6:
                    self._apply_move(two_opt_move[2], iteration + 10000)
                    improved = True

                # Try insert
                insert_move = self._try_insert(iteration + 10000)
                if insert_move is not None and insert_move[0] > 0:
                    self._apply_move(insert_move[2], iteration + 10000)
                    improved = True

                # Try replace
                replace_move = self._try_replace(iteration + 10000)
                if replace_move is not None and replace_move[0] > 0:
                    self._apply_move(replace_move[2], iteration + 10000)
                    improved = True

                if not improved:
                    break

            self._update_best()
            print(f"  Perturbation {perturbation_count}: profit={self._current_profit()}, "
                  f"best={self.best_profit}")


# ─── Main solve function ────────────────────────────────────────────────────

def solve(model, solution_file, enforce_mandatory=True):
    """
    Solve the CTOP instance:
      1. Build initial solution with greedy construction
      2. Improve with Tabu Search + ILS perturbations
      3. Write solution to file

    Parameters
    ----------
    model : Model
        Parsed CTOP instance
    solution_file : str
        Path to write the solution
    enforce_mandatory : bool
        If True, mandatory nodes must be included (Problem 2)
    """
    random.seed(seed)

    print("  Phase 1: Greedy Construction...")
    routes = greedy_construct(model, enforce_mandatory)
    init_profit = total_profit(model, routes)
    init_cost = total_cost(model, routes)
    print(f"  Initial solution: profit={init_profit}, cost={init_cost:.1f}, routes={len(routes)}")

    print("  Phase 2: Tabu Search + ILS Improvement...")
    ts = TabuSearchCTOP(model, routes, enforce_mandatory)
    routes = ts.run(max_iterations=3000, time_limit=270)

    final_profit = total_profit(model, routes)
    final_cost = total_cost(model, routes)
    print(f"  Final solution: profit={final_profit}, cost={final_cost:.1f}, routes={len(routes)}")
    print(f"  Improvement: +{final_profit - init_profit} profit, "
          f"{final_cost - init_cost:+.1f} cost")

    write_solution(routes, solution_file)
    print(f"  Solution written to {solution_file}")
