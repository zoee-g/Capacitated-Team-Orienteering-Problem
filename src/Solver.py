"""
Solver.py — CTOP Final Solver
==============================
Adaptive Greedy + ALNS (embedded Tabu) + Multi-Seed

Φάση 1: Adaptive Greedy Construction (δυναμικές ποινές)
Φάση 2: ALNS κύριος βρόχος:
    - Destroy (4): random / worst / shaw / string
    - Repair (2): greedy / regret-2
    - Embedded Tabu: Insert + Replace + 2-opt μετά κάθε repair
    - SA acceptance: δέχεται χειρότερες λύσεις φθίνουσα πιθανότητα
    - Adaptive weights: μαθαίνει ποιος operator δουλεύει
Multi-seed: 3 seeds × 85 δευτ. = 255 δευτ. < 5 λεπτά
"""

import random
import time
import math

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
    a, b = route[position - 1], route[position]
    return model.cost_matrix[a][node_id] + model.cost_matrix[node_id][b] - model.cost_matrix[a][b]

def get_all_served(routes):
    served = set()
    for r in routes:
        for n in r:
            if n != 0: served.add(n)
    return served

def write_solution(routes, solution_file):
    with open(solution_file, "w") as f:
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")

def best_insertion(model, routes, node_id):
    best = None
    for r_idx, route in enumerate(routes):
        for pos in range(1, len(route)):
            if not can_insert(model, route, node_id, pos): continue
            delta = insertion_cost_delta(model, route, node_id, pos)
            score = model.nodes[node_id].profit / (delta + 1e-6)
            if best is None or score > best[0]:
                best = (score, r_idx, pos)
    return best

def ensure_slots(routes, K):
    active = [r for r in routes if len(r) > 2]
    while len(active) < K: active.append([0, 0])
    return active


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 1: ADAPTIVE GREEDY
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_greedy_construct(model, enforce_mandatory=True):
    K, Q, T_max = model.vehicles, model.capacity, model.t_max
    N = model.num_nodes - 1
    routes = [[0, 0] for _ in range(K)]
    current_loads = [0] * K
    current_times = [0.0] * K
    available = list(range(1, N + 1))

    profits = [n.profit for n in model.nodes]
    demands = [n.demand for n in model.nodes]
    mandatory = [1 if n.isMandatory else 0 for n in model.nodes]

    while available:
        best_score, best_ins = -1, None
        for cust in available:
            ap = profits[cust]
            if enforce_mandatory and mandatory[cust] == 1:
                ap += 10 ** 9
            for v in range(K):
                tlp = (T_max - current_times[v]) / T_max if T_max > 0 else 0
                clp = (Q - current_loads[v]) / Q if Q > 0 else 0
                alpha = 1.0 / (tlp + 0.01)
                beta = 4.0 / (clp + 0.01)
                for i in range(1, len(routes[v])):
                    nl = current_loads[v] + demands[cust]
                    if nl > Q: continue
                    added = model.cost_matrix[routes[v][i-1]][cust] + model.cost_matrix[cust][routes[v][i]]
                    removed = model.cost_matrix[routes[v][i-1]][routes[v][i]]
                    ci = added - removed
                    nt = current_times[v] + ci
                    if nt > T_max: continue
                    pen = alpha * (ci / T_max) + beta * (demands[cust] / Q)
                    if pen <= 0: pen = 1e-9
                    sc = ap / pen
                    if sc > best_score:
                        best_score = sc
                        best_ins = (cust, v, i, nl, nt)
        if best_ins:
            c, v, pos, nl, nt = best_ins
            routes[v].insert(pos, c)
            current_loads[v] = nl
            current_times[v] = nt
            available.remove(c)
        else:
            break
    return [r for r in routes if len(r) > 2]


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDED TABU: γρήγορο Insert + Replace + 2-opt
# ═══════════════════════════════════════════════════════════════════════════

def quick_tabu(model, routes, mandatory_set, max_iter=100):
    """
    Γρήγορος Tabu μετά κάθε ALNS repair.
    Κάνει Insert (νέοι πελάτες) + Replace (αντικατάσταση) + 2-opt.
    ΑΥΤΟ ήταν που έλειπε από τον προηγούμενο ALNS.
    """
    K = model.vehicles
    best_routes = [r[:] for r in routes]
    best_profit = total_profit(model, routes)
    best_cost = total_cost(model, routes)

    for it in range(max_iter):
        improved = False

        # --- Insert: βάλε νέο πελάτη ---
        served = get_all_served(routes)
        unserved = [n.id for n in model.nodes if not n.isDepot and n.id not in served]
        if unserved:
            best_ins = None
            for nid in unserved:
                p = model.nodes[nid].profit
                for ri, route in enumerate(routes):
                    if route_load(model, route) + model.nodes[nid].demand > model.capacity: continue
                    for pos in range(1, len(route)):
                        if not can_insert(model, route, nid, pos): continue
                        d = insertion_cost_delta(model, route, nid, pos)
                        if best_ins is None or p > best_ins[0] or (p == best_ins[0] and d < best_ins[1]):
                            best_ins = (p, d, ri, pos, nid)
            if best_ins and best_ins[0] > 0:
                _, _, ri, pos, nid = best_ins
                routes[ri].insert(pos, nid)
                improved = True

        # --- Replace: αντικατάστασε κακό με καλό ---
        served = get_all_served(routes)
        unserved = [n.id for n in model.nodes if not n.isDepot and n.id not in served]
        best_rep = None
        for ri, route in enumerate(routes):
            for ni in range(1, len(route) - 1):
                oid = route[ni]
                if oid in mandatory_set: continue
                op = model.nodes[oid].profit
                reduced = route[:ni] + route[ni+1:]
                for nid in unserved:
                    pd = model.nodes[nid].profit - op
                    if pd <= 0: continue
                    for pos in range(1, len(reduced)):
                        test = reduced[:pos] + [nid] + reduced[pos:]
                        if route_load(model, test) > model.capacity: continue
                        if route_cost(model, test) > model.t_max: continue
                        cd = route_cost(model, test) - route_cost(model, route)
                        if best_rep is None or pd > best_rep[0] or (pd == best_rep[0] and cd < best_rep[1]):
                            best_rep = (pd, cd, ri, ni, oid, pos, nid)
        if best_rep and best_rep[0] > 0:
            _, _, ri, ni, oid, pos, nid = best_rep
            routes[ri].pop(ni)
            routes[ri].insert(pos if ni >= pos else pos - 1, nid)
            improved = True

        # --- 2-opt: φτιάξε crossings ---
        for ri, route in enumerate(routes):
            best_2opt = None
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    test = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    nc = route_cost(model, test)
                    if nc > model.t_max: continue
                    cd = nc - route_cost(model, route)
                    if cd < -1e-6 and (best_2opt is None or cd < best_2opt[0]):
                        best_2opt = (cd, i, j)
            if best_2opt:
                _, i, j = best_2opt
                routes[ri] = route[:i] + route[i:j+1][::-1] + route[j+1:]
                improved = True

        # Ενημέρωσε best
        p = total_profit(model, routes)
        c = total_cost(model, routes)
        if p > best_profit or (p == best_profit and c < best_cost):
            best_profit = p
            best_cost = c
            best_routes = [r[:] for r in routes]

        if not improved:
            break

    return best_routes


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 2: ALNS (με embedded Tabu)
# ═══════════════════════════════════════════════════════════════════════════

class ALNS_CTOP:
    SCORE_BEST = 25
    SCORE_BETTER = 9
    SCORE_ACCEPTED = 2
    SCORE_REJECTED = 0

    def __init__(self, model, routes, enforce_mandatory=True):
        self.model = model
        self.enforce_mandatory = enforce_mandatory
        self.mandatory_set = set()
        if enforce_mandatory:
            self.mandatory_set = {n.id for n in model.nodes if n.isMandatory and not n.isDepot}

        self.routes = [r[:] for r in routes]
        self.current_profit = total_profit(model, routes)
        self.current_cost = total_cost(model, routes)
        self.best_routes = [r[:] for r in routes]
        self.best_profit = self.current_profit
        self.best_cost = self.current_cost

        # Destroy operators
        self.destroy_ops = [
            ("random", self._random_removal),
            ("worst", self._worst_removal),
            ("shaw", self._shaw_removal),
            ("string", self._string_removal),
        ]
        # Repair operators
        self.repair_ops = [
            ("greedy", self._greedy_repair),
            ("regret2", self._regret_2_repair),
        ]

        # Adaptive βάρη
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.r_weights = [1.0] * len(self.repair_ops)
        self.d_scores = [0.0] * len(self.destroy_ops)
        self.r_scores = [0.0] * len(self.repair_ops)
        self.d_uses = [0] * len(self.destroy_ops)
        self.r_uses = [0] * len(self.repair_ops)

        # SA: Χρυσή Τομή exploration → exploitation
        # T=500: δέχεται 55% χειρότερες λύσεις (exploration)
        # T=10:  δέχεται <1% χειρότερες λύσεις (exploitation)
        # Cooling 0.992: 500→10 σε ~500 iterations
        self.sa_temp = 500.0
        self.sa_cooling = 0.992
        self.segment_size = 50
        self.weight_decay = 0.8

    # ── DESTROY ──────────────────────────────────────────────────────

    def _removable(self, routes):
        return [n for r in routes for n in r if n != 0 and n not in self.mandatory_set]

    def _random_removal(self, routes):
        served = self._removable(routes)
        if not served: return routes, []
        num = max(3, len(served) // 5)
        removed = random.sample(served, min(num, len(served)))
        for nid in removed:
            for r in routes:
                if nid in r: r.remove(nid); break
        return routes, removed

    def _worst_removal(self, routes):
        m = self.model
        scores = []
        for ri, route in enumerate(routes):
            for ni in range(1, len(route) - 1):
                nid = route[ni]
                if nid in self.mandatory_set: continue
                prev, succ = route[ni-1], route[ni+1]
                cost_c = m.cost_matrix[prev][nid] + m.cost_matrix[nid][succ] - m.cost_matrix[prev][succ]
                profit = m.nodes[nid].profit
                scores.append((profit / (cost_c + 1e-6), nid))
        scores.sort()
        removed = []
        for i in range(min(5, len(scores))):
            nid = scores[i][1]
            for r in routes:
                if nid in r: r.remove(nid); removed.append(nid); break
        return routes, removed

    def _shaw_removal(self, routes):
        m = self.model
        served = self._removable(routes)
        if not served: return routes, []
        num = max(3, len(served) // 7)
        seed_n = random.choice(served)
        to_remove = [seed_n]
        while len(to_remove) < num:
            ref = random.choice(to_remove)
            remaining = [c for c in served if c not in to_remove]
            if not remaining: break
            remaining.sort(key=lambda c: m.cost_matrix[ref][c] + abs(m.nodes[ref].demand - m.nodes[c].demand))
            to_remove.append(remaining[0])
        for nid in to_remove:
            for r in routes:
                if nid in r: r.remove(nid); break
        return routes, to_remove

    def _string_removal(self, routes):
        eligible = [(ri, r) for ri, r in enumerate(routes) if len(r) > 4]
        if not eligible: return self._random_removal(routes)
        ri, route = random.choice(eligible)
        max_size = max(2, (len(route) - 2) // 3)
        size = random.randint(2, max_size)
        start = random.randint(1, len(route) - 1 - size)
        removed = []
        for idx in sorted(range(start, start + size), reverse=True):
            nid = route[idx]
            if nid != 0 and nid not in self.mandatory_set:
                route.pop(idx); removed.append(nid)
        return routes, removed

    # ── REPAIR ───────────────────────────────────────────────────────

    def _greedy_repair(self, routes, unassigned):
        m = self.model
        unassigned.sort(key=lambda nid: m.nodes[nid].profit, reverse=True)
        for nid in unassigned:
            result = best_insertion(m, routes, nid)
            if result: _, ri, pos = result; routes[ri].insert(pos, nid)
        return routes

    def _regret_2_repair(self, routes, unassigned):
        m = self.model
        remaining = list(unassigned)
        while remaining:
            regrets = []
            for nid in remaining:
                insertions = []
                for ri, route in enumerate(routes):
                    if route_load(m, route) + m.nodes[nid].demand > m.capacity: continue
                    for pos in range(1, len(route)):
                        if not can_insert(m, route, nid, pos): continue
                        delta = insertion_cost_delta(m, route, nid, pos)
                        insertions.append((m.nodes[nid].profit - delta, ri, pos))
                if not insertions:
                    regrets.append((-1e9, nid, None)); continue
                insertions.sort(key=lambda x: x[0], reverse=True)
                best_s = insertions[0][0]
                second_s = insertions[1][0] if len(insertions) > 1 else best_s - 100
                regrets.append((best_s - second_s, nid, insertions[0]))
            regrets.sort(key=lambda x: x[0], reverse=True)
            _, nid, best_move = regrets[0]
            if best_move:
                _, ri, pos = best_move
                routes[ri].insert(pos, nid)
            remaining.remove(nid)
        return routes

    # ── ROULETTE + SA + WEIGHTS ──────────────────────────────────────

    def _roulette(self, weights):
        total = sum(weights)
        r = random.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum: return i
        return len(weights) - 1

    def _sa_accept(self, new_profit):
        if new_profit >= self.current_profit: return True
        delta = self.current_profit - new_profit
        if self.sa_temp > 0.01:
            return random.random() < math.exp(-delta / self.sa_temp)
        return False

    def _update_weights(self):
        for i in range(len(self.d_weights)):
            if self.d_uses[i] > 0:
                avg = self.d_scores[i] / self.d_uses[i]
                self.d_weights[i] = self.weight_decay * self.d_weights[i] + (1 - self.weight_decay) * avg
                self.d_weights[i] = max(self.d_weights[i], 0.1)
            self.d_scores[i] = 0; self.d_uses[i] = 0
        for i in range(len(self.r_weights)):
            if self.r_uses[i] > 0:
                avg = self.r_scores[i] / self.r_uses[i]
                self.r_weights[i] = self.weight_decay * self.r_weights[i] + (1 - self.weight_decay) * avg
                self.r_weights[i] = max(self.r_weights[i], 0.1)
            self.r_scores[i] = 0; self.r_uses[i] = 0

    # ── ΚΥΡΙΟΣ ΒΡΟΧΟΣ ───────────────────────────────────────────────

    def run(self, time_limit_sec=85):
        t0 = time.time()
        iteration = 0

        while time.time() - t0 < time_limit_sec:
            iteration += 1

            # 1. Επιλογή operators
            d_idx = self._roulette(self.d_weights)
            r_idx = self._roulette(self.r_weights)
            d_name, d_func = self.destroy_ops[d_idx]
            r_name, r_func = self.repair_ops[r_idx]

            # 2. Αντίγραφο
            new_routes = [r[:] for r in self.routes]
            new_routes = ensure_slots(new_routes, self.model.vehicles)

            # 3. DESTROY
            new_routes, removed = d_func(new_routes)
            new_routes = ensure_slots(new_routes, self.model.vehicles)

            # 4. REPAIR
            new_routes = r_func(new_routes, removed)

            # 5. EMBEDDED TABU ← ΤΟ ΚΛΕΙΔΙ
            #    Μετά το repair, τρέξε γρήγορο Tabu (Insert + Replace + 2-opt)
            new_routes = [r for r in new_routes if len(r) > 2]
            new_routes = ensure_slots(new_routes, self.model.vehicles)
            new_routes = quick_tabu(self.model, new_routes, self.mandatory_set, max_iter=30)
            new_routes = [r for r in new_routes if len(r) > 2]

            # 6. Αξιολόγηση
            new_p = total_profit(self.model, new_routes)
            new_c = total_cost(self.model, new_routes)

            # Έλεγχος εγκυρότητας
            valid = True
            for r in new_routes:
                if route_load(self.model, r) > self.model.capacity or route_cost(self.model, r) > self.model.t_max:
                    valid = False; break
            if valid and self.enforce_mandatory:
                if not self.mandatory_set.issubset(get_all_served(new_routes)):
                    valid = False

            # 7. Βαθμολόγηση + αποδοχή
            if valid:
                is_best = (new_p > self.best_profit or
                           (new_p == self.best_profit and new_c < self.best_cost))
                is_better = (new_p > self.current_profit or
                             (new_p == self.current_profit and new_c < self.current_cost))
                is_accepted = self._sa_accept(new_p)

                if is_best:
                    score = self.SCORE_BEST
                    self.best_routes = [r[:] for r in new_routes]
                    self.best_profit = new_p; self.best_cost = new_c
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_p; self.current_cost = new_c
                elif is_better:
                    score = self.SCORE_BETTER
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_p; self.current_cost = new_c
                elif is_accepted:
                    score = self.SCORE_ACCEPTED
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_p; self.current_cost = new_c
                else:
                    score = self.SCORE_REJECTED
            else:
                score = self.SCORE_REJECTED

            # 8. Scores
            self.d_scores[d_idx] += score; self.d_uses[d_idx] += 1
            self.r_scores[r_idx] += score; self.r_uses[r_idx] += 1

            # 9. Ενημέρωση βαρών
            if iteration % self.segment_size == 0:
                self._update_weights()

            # 10. SA cooling
            self.sa_temp *= self.sa_cooling

            if iteration % 100 == 0:
                elapsed = time.time() - t0
                print(f"    ALNS iter {iteration}: profit={self.current_profit}, "
                      f"best={self.best_profit}, T={self.sa_temp:.2f}, t={elapsed:.0f}s")

        print(f"    ALNS done: {iteration} iters, best={self.best_profit}")
        return self.best_routes, self.best_profit, self.best_cost


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN: MULTI-SEED
# ═══════════════════════════════════════════════════════════════════════════

def solve(model, solution_file, enforce_mandatory=True):
    """3 seeds × 85 δευτ. = 255 δευτ. < 5 λεπτά."""
    best_seeds = [42, 15, 23]  # 3 seeds αρκούν
    time_per_seed = 85

    overall_best_routes = None
    overall_best_profit = -1
    overall_best_cost = float('inf')

    for s in best_seeds:
        random.seed(s)
        print(f"\n  === Seed {s} ===")

        routes = adaptive_greedy_construct(model, enforce_mandatory)
        init_p = total_profit(model, routes)
        print(f"  Greedy: profit={init_p}")

        alns = ALNS_CTOP(model, routes, enforce_mandatory)
        routes, p, c = alns.run(time_limit_sec=time_per_seed)

        print(f"  Seed {s}: profit={p}, cost={c:.1f}")

        if p > overall_best_profit or (p == overall_best_profit and c < overall_best_cost):
            overall_best_profit = p
            overall_best_cost = c
            overall_best_routes = [r[:] for r in routes]
            print(f"  ★ New best!")

    print(f"\n  === FINAL: profit={overall_best_profit}, cost={overall_best_cost:.1f} ===")
    write_solution(overall_best_routes, solution_file)
    print(f"  Solution written to {solution_file}")