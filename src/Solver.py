"""
Solver.py — CTOP Solver (Adaptive Greedy + Tabu Search + ILS)
==============================================================
Φάση 1: Adaptive Weighted Greedy (καλύτερη αρχική λύση)
Φάση 2: Tabu Search + ILS (αποδεδειγμένη βελτίωση)

Seed: 42
"""

import random
import time

seed = 42
random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
#  ΒΟΗΘΗΤΙΚΕΣ
# ═══════════════════════════════════════════════════════════════════════════

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

def can_insert(model, route, node_id, position):
    new_route = route[:position] + [node_id] + route[position:]
    return (route_load(model, new_route) <= model.capacity and
            route_cost(model, new_route) <= model.t_max)

def insertion_cost_delta(model, route, node_id, position):
    a = route[position - 1]
    b = route[position]
    return (model.cost_matrix[a][node_id] +
            model.cost_matrix[node_id][b] -
            model.cost_matrix[a][b])

def get_all_served(routes):
    served = set()
    for r in routes:
        for n in r:
            if n != 0:
                served.add(n)
    return served

def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 1: ADAPTIVE WEIGHTED GREEDY
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_greedy_construct(model, enforce_mandatory=True):
    K = model.vehicles
    Q = model.capacity
    T_max = model.t_max
    N = model.num_nodes - 1

    routes = [[0, 0] for _ in range(K)]
    current_loads = [0] * K
    current_times = [0.0] * K

    available = list(range(1, N + 1))
    TIME_WEIGHT = 1.0
    CAPACITY_WEIGHT = 4.0

    profits = [node.profit for node in model.nodes]
    demands = [node.demand for node in model.nodes]
    mandatory = [1 if node.isMandatory else 0 for node in model.nodes]

    while available:
        best_score = -1
        best_ins = None

        for cust in available:
            actual_profit = profits[cust]
            if enforce_mandatory and mandatory[cust] == 1:
                actual_profit += 10 ** 9

            for v in range(K):
                time_left_pct = (T_max - current_times[v]) / T_max if T_max > 0 else 0
                cap_left_pct = (Q - current_loads[v]) / Q if Q > 0 else 0
                alpha = TIME_WEIGHT / (time_left_pct + 0.01)
                beta = CAPACITY_WEIGHT / (cap_left_pct + 0.01)

                for i in range(1, len(routes[v])):
                    prev_node = routes[v][i - 1]
                    next_node = routes[v][i]
                    new_load = current_loads[v] + demands[cust]
                    if new_load > Q:
                        continue
                    added = model.cost_matrix[prev_node][cust] + model.cost_matrix[cust][next_node]
                    removed = model.cost_matrix[prev_node][next_node]
                    cost_increase = added - removed
                    new_time = current_times[v] + cost_increase
                    if new_time > T_max:
                        continue
                    norm_time = cost_increase / T_max if T_max > 0 else 0
                    norm_demand = demands[cust] / Q if Q > 0 else 0
                    penalty = alpha * norm_time + beta * norm_demand
                    if penalty <= 0:
                        penalty = 1e-9
                    score = actual_profit / penalty
                    if score > best_score:
                        best_score = score
                        best_ins = (cust, v, i, new_load, new_time)

        if best_ins:
            cust, v, pos, n_load, n_time = best_ins
            routes[v].insert(pos, cust)
            current_loads[v] = n_load
            current_times[v] = n_time
            available.remove(cust)
        else:
            break

    routes = [r for r in routes if len(r) > 2]
    return routes


# ═══════════════════════════════════════════════════════════════════════════
#  ΒΟΗΘΗΤΙΚΗ: Best Insertion (για ILS re-insertion)
# ═══════════════════════════════════════════════════════════════════════════

def best_insertion(model, routes, node_id):
    best = None
    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            if not can_insert(model, route, node_id, pos):
                continue
            delta = insertion_cost_delta(model, route, node_id, pos)
            score = model.nodes[node_id].profit / (delta + 1e-6)
            if best is None or score > best[0]:
                best = (score, r_idx, pos)
    return best


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 2: TABU SEARCH + ILS
# ═══════════════════════════════════════════════════════════════════════════

class TabuSearchCTOP:

    def __init__(self, model, routes, enforce_mandatory=True):
        self.model = model
        self.routes = [r[:] for r in routes]
        self.enforce_mandatory = enforce_mandatory
        self.mandatory_set = set()
        if enforce_mandatory:
            self.mandatory_set = {n.id for n in model.nodes if n.isMandatory and not n.isDepot}
        self.tabu = {}
        self.min_tenure = 5
        self.max_tenure = 15
        self.best_routes = None
        self.best_profit = -1
        self.best_time = float('inf')
        self._update_best()

    def _served(self):
        return get_all_served(self.routes)

    def _unserved(self):
        return {n.id for n in self.model.nodes if not n.isDepot} - self._served()

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
        return node_id in self.mandatory_set

    # --- Operator 1: Relocate ---
    def _try_relocate(self, iteration):
        best_move = None
        m = self.model
        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                node_id = route[ni]
                if self._is_tabu(node_id, iteration):
                    continue
                for rj, target in enumerate(self.routes):
                    for pj in range(1, len(target)):
                        if ri == rj and pj in (ni, ni + 1):
                            continue
                        if ri == rj:
                            test = route[:ni] + route[ni + 1:]
                            adj_pos = pj if ni > pj else pj - 1
                            test = test[:adj_pos] + [node_id] + test[adj_pos:]
                        else:
                            test = target[:pj] + [node_id] + target[pj:]
                        if route_load(m, test) > m.capacity or route_cost(m, test) > m.t_max:
                            continue
                        if ri == rj:
                            cost_delta = route_cost(m, test) - route_cost(m, route)
                        else:
                            cost_delta = ((route_cost(m, route[:ni] + route[ni + 1:]) + route_cost(m, test))
                                          - (route_cost(m, route) + route_cost(m, target)))
                        if best_move is None or cost_delta < best_move[1]:
                            best_move = (0, cost_delta, ('relocate', ri, ni, rj, pj, node_id))
        return best_move

    # --- Operator 2: Swap ---
    def _try_swap(self, iteration):
        best_move = None
        m = self.model
        for ri in range(len(self.routes)):
            r1 = self.routes[ri]
            for rj in range(ri, len(self.routes)):
                r2 = self.routes[rj]
                for ni in range(1, len(r1) - 1):
                    start_nj = ni + 1 if ri == rj else 1
                    for nj in range(start_nj, len(r2) - 1):
                        n1, n2 = r1[ni], r2[nj]
                        if self._is_tabu(n1, iteration) or self._is_tabu(n2, iteration):
                            continue
                        if ri == rj:
                            test = r1[:]
                            test[ni], test[nj] = n2, n1
                            if route_load(m, test) > m.capacity or route_cost(m, test) > m.t_max:
                                continue
                            cost_delta = route_cost(m, test) - route_cost(m, r1)
                        else:
                            t1, t2 = r1[:], r2[:]
                            t1[ni], t2[nj] = n2, n1
                            if (route_load(m, t1) > m.capacity or route_load(m, t2) > m.capacity or
                                    route_cost(m, t1) > m.t_max or route_cost(m, t2) > m.t_max):
                                continue
                            cost_delta = ((route_cost(m, t1) + route_cost(m, t2))
                                          - (route_cost(m, r1) + route_cost(m, r2)))
                        if best_move is None or cost_delta < best_move[1]:
                            best_move = (0, cost_delta, ('swap', ri, ni, rj, nj))
        return best_move

    # --- Operator 3: 2-opt ---
    def _try_two_opt(self, iteration):
        best_move = None
        m = self.model
        for ri, route in enumerate(self.routes):
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    test = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    new_cost = route_cost(m, test)
                    if new_cost > m.t_max:
                        continue
                    cost_delta = new_cost - route_cost(m, route)
                    if best_move is None or cost_delta < best_move[1]:
                        best_move = (0, cost_delta, ('2opt', ri, i, j))
        return best_move

    # --- Operator 4: Insert ---
    def _try_insert(self, iteration):
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
                    if best_move is None or profit > best_move[0] or (profit == best_move[0] and delta < best_move[1]):
                        best_move = (profit, delta, ('insert', ri, pos, node_id))
        return best_move

    # --- Operator 5: Replace ---
    def _try_replace(self, iteration):
        best_move = None
        m = self.model
        unserved = self._unserved()
        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                old_id = route[ni]
                if self._is_protected(old_id) or self._is_tabu(old_id, iteration):
                    continue
                old_profit = m.nodes[old_id].profit
                reduced = route[:ni] + route[ni + 1:]
                for new_id in unserved:
                    profit_delta = m.nodes[new_id].profit - old_profit
                    if profit_delta <= 0 or self._is_tabu(new_id, iteration):
                        continue
                    for pos in range(1, len(reduced)):
                        test = reduced[:pos] + [new_id] + reduced[pos:]
                        if route_load(m, test) > m.capacity or route_cost(m, test) > m.t_max:
                            continue
                        cost_delta = route_cost(m, test) - route_cost(m, route)
                        if best_move is None or profit_delta > best_move[0] or (
                                profit_delta == best_move[0] and cost_delta < best_move[1]):
                            best_move = (profit_delta, cost_delta, ('replace', ri, ni, old_id, pos, new_id))
        return best_move

    # --- Apply ---
    def _apply_move(self, move_info, iteration):
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
            adj_pos = pos if ni >= pos else pos - 1
            self.routes[ri].insert(adj_pos, new_id)
            self._set_tabu(old_id, iteration)
            self._set_tabu(new_id, iteration)
        self.routes = [r for r in self.routes if len(r) > 2]
        while len(self.routes) < self.model.vehicles:
            self.routes.append([0, 0])

    # --- Main Tabu + ILS loop ---
    def run(self, max_iterations=3000, time_limit=270):
        start_time = time.time()
        no_improve = 0
        max_no_improve = 300

        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                print(f"  Time limit reached at iteration {iteration}")
                break

            candidates = []
            insert_move = self._try_insert(iteration)
            if insert_move is not None:
                candidates.append(insert_move)
            replace_move = self._try_replace(iteration)
            if replace_move is not None:
                candidates.append(replace_move)
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

            candidates.sort(key=lambda x: (-x[0], x[1]))
            best = candidates[0]

            if best[0] > 0 or best[1] < -1e-6:
                self._apply_move(best[2], iteration)
                self._update_best()
                no_improve = 0
                if iteration % 100 == 0:
                    print(f"  Iter {iteration}: profit={self._current_profit()}, "
                          f"cost={self._current_cost():.1f}, best_profit={self.best_profit}")
            else:
                no_improve += 1
                if best[0] == 0 and no_improve < max_no_improve:
                    self._apply_move(best[2], iteration)
                if no_improve > max_no_improve:
                    break

        # ILS Perturbation phase
        self._perturbation_phase(start_time, time_limit)
        self.routes = [r for r in self.best_routes if len(r) > 2]
        return self.routes

    def _perturbation_phase(self, start_time, time_limit):
        perturbation_count = 0
        while time.time() - start_time < time_limit - 10:
            perturbation_count += 1
            self.routes = [r[:] for r in self.best_routes]

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

            self.routes = [r for r in self.routes if len(r) > 2]
            while len(self.routes) < self.model.vehicles:
                self.routes.append([0, 0])

            all_candidates = list(to_remove) + list(self._unserved())
            random.shuffle(all_candidates)
            all_candidates.sort(key=lambda i: self.model.nodes[i].profit, reverse=True)

            for node_id in all_candidates:
                if node_id in self._served():
                    continue
                result = best_insertion(self.model, self.routes, node_id)
                if result is not None:
                    _, r_idx, pos = result
                    self.routes[r_idx].insert(pos, node_id)

            self.tabu.clear()

            for iteration in range(500):
                if time.time() - start_time > time_limit - 5:
                    break
                improved = False

                two_opt_move = self._try_two_opt(iteration + 10000)
                if two_opt_move is not None and two_opt_move[1] < -1e-6:
                    self._apply_move(two_opt_move[2], iteration + 10000)
                    improved = True

                insert_move = self._try_insert(iteration + 10000)
                if insert_move is not None and insert_move[0] > 0:
                    self._apply_move(insert_move[2], iteration + 10000)
                    improved = True

                replace_move = self._try_replace(iteration + 10000)
                if replace_move is not None and replace_move[0] > 0:
                    self._apply_move(replace_move[2], iteration + 10000)
                    improved = True

                if not improved:
                    break

            self._update_best()
            if perturbation_count % 10 == 0:
                print(f"  Perturbation {perturbation_count}: profit={self._current_profit()}, "
                      f"best={self.best_profit}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def solve(model, solution_file, enforce_mandatory=True):
    random.seed(seed)

    print("  Phase 1: Adaptive Greedy Construction...")
    routes = adaptive_greedy_construct(model, enforce_mandatory)
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