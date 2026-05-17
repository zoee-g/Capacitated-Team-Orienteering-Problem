"""
Solver.py — CTOP Solver (Adaptive Greedy + Tabu Search + ALNS)
==============================================================
Αρχιτεκτονική 3 Φάσεων:

Φάση 1: Adaptive Weighted Greedy Construction
    Κατασκευή αρχικής εφικτής λύσης με δυναμικές ποινές
    που προσαρμόζονται στον εναπομείναντα χρόνο/χωρητικότητα.

Φάση 2: Tabu Search
    Τοπική αναζήτηση με 5 τελεστές (Relocate, Swap, 2-opt,
    Insert, Replace). Tabu list αποτρέπει κύκλους.

Φάση 3: ALNS (Adaptive Large Neighborhood Search)
    Μεγάλης κλίμακας βελτίωση με πολλαπλές στρατηγικές
    καταστροφής/ανακατασκευής και προσαρμοστικά βάρη.

    Destroy operators (4):
      - random_removal:  τυχαία αφαίρεση ~15% πελατών
      - worst_removal:   αφαίρεση πελατών με χειρότερο profit/cost
      - shaw_removal:    αφαίρεση γεωγραφικά παρόμοιων πελατών
      - string_removal:  αφαίρεση τμήματος συνεχόμενων πελατών

    Repair operators (2):
      - greedy_repair:   εισαγωγή στη φθηνότερη εφικτή θέση
      - regret_2_repair: εισαγωγή πρώτα αυτών που θα "μετανιώσουμε"

    Adaptive βάρη: operators που βρίσκουν καλές λύσεις
    επιλέγονται πιο συχνά (Roulette Wheel Selection).

    Αποδοχή χειρότερων: Simulated Annealing (φθίνουσα θερμοκρασία).

Στόχος: Μεγιστοποίηση κέρδους. Ισοβαθμία: ελαχιστοποίηση χρόνου.
Seed: 42
"""

import random
import time
import math

seed = 42
random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════
#  ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ═══════════════════════════════════════════════════════════════════════════

def route_cost(model, route):
    """Συνολικός χρόνος μιας διαδρομής."""
    return sum(model.cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def route_load(model, route):
    """Συνολικός όγκος φορτίου μιας διαδρομής."""
    return sum(model.nodes[i].demand for i in route if i != 0)


def route_profit(model, route):
    """Συνολικό κέρδος μιας διαδρομής."""
    return sum(model.nodes[i].profit for i in route if i != 0)


def total_profit(model, routes):
    """Συνολικό κέρδος όλων των διαδρομών."""
    return sum(route_profit(model, r) for r in routes)


def total_cost(model, routes):
    """Συνολικός χρόνος όλων των διαδρομών."""
    return sum(route_cost(model, r) for r in routes)


def can_insert(model, route, node_id, position):
    """Ελέγχει αν χωράει ο node_id στη θέση position."""
    new_route = route[:position] + [node_id] + route[position:]
    return (route_load(model, new_route) <= model.capacity and
            route_cost(model, new_route) <= model.t_max)


def insertion_cost_delta(model, route, node_id, position):
    """Αύξηση κόστους από εισαγωγή node_id στη θέση position."""
    a = route[position - 1]
    b = route[position]
    return (model.cost_matrix[a][node_id] +
            model.cost_matrix[node_id][b] -
            model.cost_matrix[a][b])


def get_all_served(routes):
    """Σύνολο εξυπηρετούμενων πελατών (χωρίς depot)."""
    served = set()
    for r in routes:
        for n in r:
            if n != 0:
                served.add(n)
    return served


def write_solution(routes, solution_file):
    """Γράφει τη λύση στο αρχείο."""
    with open(solution_file, "w") as f:
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 1: ADAPTIVE WEIGHTED GREEDY CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_greedy_construct(model, enforce_mandatory=True):
    """
    Κατασκευή αρχικής λύσης με δυναμικές ποινές.
    score = profit / (alpha * norm_time + beta * norm_demand)
    alpha, beta αυξάνονται καθώς γεμίζει το όχημα.
    """
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
#  ΦΑΣΗ 2: TABU SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def best_insertion(model, routes, node_id):
    """Βρίσκει καλύτερη θέση εισαγωγής (score = profit / cost_increase)."""
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


class TabuSearchCTOP:
    """Tabu Search με 5 τελεστές για βελτίωση κέρδους CTOP."""

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
            return True
        return False

    def _is_tabu(self, node_id, it):
        return self.tabu.get(node_id, -1) > it

    def _set_tabu(self, node_id, it):
        self.tabu[node_id] = it + random.randint(self.min_tenure, self.max_tenure)

    def _is_protected(self, nid):
        return nid in self.mandatory_set

    def _ensure_slots(self):
        active = [r for r in self.routes if len(r) > 2]
        self.routes = active
        while len(self.routes) < self.model.vehicles:
            self.routes.append([0, 0])

    def _try_relocate(self, it):
        best = None
        m = self.model
        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                nid = route[ni]
                if self._is_tabu(nid, it): continue
                for rj, target in enumerate(self.routes):
                    for pj in range(1, len(target)):
                        if ri == rj and pj in (ni, ni + 1): continue
                        if ri == rj:
                            test = route[:ni] + route[ni + 1:]
                            adj = pj if ni > pj else pj - 1
                            test = test[:adj] + [nid] + test[adj:]
                        else:
                            test = target[:pj] + [nid] + target[pj:]
                        if route_load(m, test) > m.capacity or route_cost(m, test) > m.t_max: continue
                        if ri == rj:
                            cd = route_cost(m, test) - route_cost(m, route)
                        else:
                            cd = ((route_cost(m, route[:ni] + route[ni + 1:]) + route_cost(m, test))
                                  - (route_cost(m, route) + route_cost(m, target)))
                        if best is None or cd < best[1]:
                            best = (0, cd, ('relocate', ri, ni, rj, pj, nid))
        return best

    def _try_swap(self, it):
        best = None
        m = self.model
        for ri in range(len(self.routes)):
            r1 = self.routes[ri]
            for rj in range(ri, len(self.routes)):
                r2 = self.routes[rj]
                for ni in range(1, len(r1) - 1):
                    sj = ni + 1 if ri == rj else 1
                    for nj in range(sj, len(r2) - 1):
                        if self._is_tabu(r1[ni], it) or self._is_tabu(r2[nj], it): continue
                        if ri == rj:
                            t = r1[:];
                            t[ni], t[nj] = r2[nj], r1[ni]
                            if route_load(m, t) > m.capacity or route_cost(m, t) > m.t_max: continue
                            cd = route_cost(m, t) - route_cost(m, r1)
                        else:
                            t1, t2 = r1[:], r2[:]
                            t1[ni], t2[nj] = r2[nj], r1[ni]
                            if (route_load(m, t1) > m.capacity or route_load(m, t2) > m.capacity or
                                    route_cost(m, t1) > m.t_max or route_cost(m, t2) > m.t_max): continue
                            cd = (route_cost(m, t1) + route_cost(m, t2)) - (route_cost(m, r1) + route_cost(m, r2))
                        if best is None or cd < best[1]:
                            best = (0, cd, ('swap', ri, ni, rj, nj))
        return best

    def _try_two_opt(self, it):
        best = None
        m = self.model
        for ri, route in enumerate(self.routes):
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    test = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    nc = route_cost(m, test)
                    if nc > m.t_max: continue
                    cd = nc - route_cost(m, route)
                    if best is None or cd < best[1]:
                        best = (0, cd, ('2opt', ri, i, j))
        return best

    def _try_insert(self, it):
        best = None
        m = self.model
        for nid in self._unserved():
            if self._is_tabu(nid, it): continue
            p = m.nodes[nid].profit
            for ri, route in enumerate(self.routes):
                if route_load(m, route) + m.nodes[nid].demand > m.capacity: continue
                for pos in range(1, len(route)):
                    if not can_insert(m, route, nid, pos): continue
                    d = insertion_cost_delta(m, route, nid, pos)
                    if best is None or p > best[0] or (p == best[0] and d < best[1]):
                        best = (p, d, ('insert', ri, pos, nid))
        return best

    def _try_replace(self, it):
        best = None
        m = self.model
        unserved = self._unserved()
        for ri, route in enumerate(self.routes):
            for ni in range(1, len(route) - 1):
                oid = route[ni]
                if self._is_protected(oid) or self._is_tabu(oid, it): continue
                op = m.nodes[oid].profit
                reduced = route[:ni] + route[ni + 1:]
                for nid in unserved:
                    pd = m.nodes[nid].profit - op
                    if pd <= 0 or self._is_tabu(nid, it): continue
                    for pos in range(1, len(reduced)):
                        test = reduced[:pos] + [nid] + reduced[pos:]
                        if route_load(m, test) > m.capacity or route_cost(m, test) > m.t_max: continue
                        cd = route_cost(m, test) - route_cost(m, route)
                        if best is None or pd > best[0] or (pd == best[0] and cd < best[1]):
                            best = (pd, cd, ('replace', ri, ni, oid, pos, nid))
        return best

    def _apply_move(self, info, it):
        kind = info[0]
        if kind == 'relocate':
            _, ri, ni, rj, pj, nid = info
            if ri == rj:
                r = self.routes[ri];
                r.pop(ni)
                r.insert(pj if ni > pj else pj - 1, nid)
            else:
                self.routes[ri].pop(ni);
                self.routes[rj].insert(pj, nid)
            self._set_tabu(nid, it)
        elif kind == 'swap':
            _, ri, ni, rj, nj = info
            if ri == rj:
                self.routes[ri][ni], self.routes[ri][nj] = self.routes[ri][nj], self.routes[ri][ni]
            else:
                self.routes[ri][ni], self.routes[rj][nj] = self.routes[rj][nj], self.routes[ri][ni]
            self._set_tabu(self.routes[ri][ni], it);
            self._set_tabu(self.routes[rj][nj], it)
        elif kind == '2opt':
            _, ri, i, j = info
            self.routes[ri] = self.routes[ri][:i] + self.routes[ri][i:j + 1][::-1] + self.routes[ri][j + 1:]
        elif kind == 'insert':
            _, ri, pos, nid = info
            self.routes[ri].insert(pos, nid);
            self._set_tabu(nid, it)
        elif kind == 'replace':
            _, ri, ni, oid, pos, nid = info
            self.routes[ri].pop(ni)
            self.routes[ri].insert(pos if ni >= pos else pos - 1, nid)
            self._set_tabu(oid, it);
            self._set_tabu(nid, it)
        self._ensure_slots()

    def run(self, max_iter=2000, time_limit_sec=80):
        """Κύριος βρόχος Tabu Search. Κρατάει ~30% του χρόνου."""
        t0 = time.time()
        no_imp = 0
        for it in range(max_iter):
            if time.time() - t0 > time_limit_sec: break
            cands = []
            for fn in [self._try_insert, self._try_replace, self._try_relocate, self._try_swap, self._try_two_opt]:
                r = fn(it)
                if r is not None: cands.append(r)
            if not cands:
                no_imp += 1
                if no_imp > 200: break
                continue
            cands.sort(key=lambda x: (-x[0], x[1]))
            best = cands[0]
            if best[0] > 0 or best[1] < -1e-6:
                self._apply_move(best[2], it)
                if self._update_best():
                    no_imp = 0
                else:
                    no_imp += 1
            else:
                no_imp += 1
                if best[0] == 0 and no_imp < 200:
                    self._apply_move(best[2], it)
            if no_imp > 200: break
            if it % 300 == 0:
                print(f"    Tabu iter {it}: profit={self._current_profit()}, best={self.best_profit}")
        return self.best_routes, self.best_profit, self.best_time


# ═══════════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 3: ALNS (Adaptive Large Neighborhood Search)
# ═══════════════════════════════════════════════════════════════════════════

class ALNS_CTOP:
    """
    ALNS προσαρμοσμένος για CTOP.

    Σε κάθε iteration:
      1. Επιλέγεται ένας destroy operator (Roulette Wheel βάσει βαρών)
      2. Επιλέγεται ένας repair operator (Roulette Wheel βάσει βαρών)
      3. Εφαρμόζεται destroy → αφαιρούνται πελάτες
      4. Εφαρμόζεται repair → εισάγονται πελάτες (στοχεύοντας μέγιστο κέρδος)
      5. Αν η νέα λύση γίνεται αποδεκτή (Simulated Annealing):
           - Αν νέο best → +25 πόντοι στους operators
           - Αν καλύτερη από τρέχουσα → +9 πόντοι
           - Αν αποδεκτή (χειρότερη αλλά SA λέει OK) → +2 πόντοι
         Αλλιώς → +0 πόντοι
      6. Κάθε segment (50 iterations) ενημερώνονται τα βάρη

    Adaptation for CTOP vs CVRP:
      - Objective: MAXIMIZE profit (αντί minimize cost)
      - Δεν χρειάζεται να εξυπηρετηθούν όλοι
      - Mandatory nodes δεν αφαιρούνται ποτέ
      - Repair στοχεύει profit/cost ratio (αντί μόνο cost)
    """

    # ── Βαθμολογίες (scores) για ενημέρωση βαρών ─────────────────
    SCORE_BEST = 25  # βρήκε νέο global best
    SCORE_BETTER = 9  # βρήκε λύση καλύτερη από τρέχουσα
    SCORE_ACCEPTED = 2  # SA αποδέχτηκε χειρότερη λύση
    SCORE_REJECTED = 0  # λύση απορρίφθηκε

    def __init__(self, model, routes, enforce_mandatory=True):
        self.model = model
        self.enforce_mandatory = enforce_mandatory
        self.mandatory_set = set()
        if enforce_mandatory:
            self.mandatory_set = {n.id for n in model.nodes if n.isMandatory and not n.isDepot}

        # Τρέχουσα λύση
        self.routes = [r[:] for r in routes]
        self.current_profit = total_profit(model, routes)
        self.current_cost = total_cost(model, routes)

        # Καλύτερη λύση
        self.best_routes = [r[:] for r in routes]
        self.best_profit = self.current_profit
        self.best_cost = self.current_cost

        # ── Destroy operators ──────────────────────────────────────
        self.destroy_ops = [
            ("random_removal", self._random_removal),
            ("worst_removal", self._worst_removal),
            ("shaw_removal", self._shaw_removal),
            ("string_removal", self._string_removal),
        ]

        # ── Repair operators ───────────────────────────────────────
        self.repair_ops = [
            ("greedy_repair", self._greedy_repair),
            ("regret_2_repair", self._regret_2_repair),
        ]

        # ── Adaptive βάρη (αρχικοποιούνται ίσα) ───────────────────
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)
        self.destroy_scores = [0.0] * len(self.destroy_ops)
        self.repair_scores = [0.0] * len(self.repair_ops)
        self.destroy_uses = [0] * len(self.destroy_ops)
        self.repair_uses = [0] * len(self.repair_ops)

        # ── Simulated Annealing παράμετροι ─────────────────────────
        self.sa_temperature = 50.0
        self.sa_cooling = 0.995  # θερμοκρασία *= cooling κάθε iteration

        # ── ALNS παράμετροι ────────────────────────────────────────
        self.segment_size = 50  # κάθε πόσα iterations ενημερώνονται βάρη
        self.weight_decay = 0.8  # decay παλαιών βαρών

    # ── DESTROY OPERATORS ────────────────────────────────────────────

    def _random_removal(self, routes):
        """
        Random Removal: αφαιρεί ~15% τυχαίους πελάτες.
        Μέγιστη τυχαιότητα → diversification.
        """
        served = [n for r in routes for n in r
                  if n != 0 and n not in self.mandatory_set]
        if not served:
            return routes, []
        num = max(3, int(len(served) * 0.15))
        removed = random.sample(served, min(num, len(served)))
        for nid in removed:
            for r in routes:
                if nid in r:
                    r.remove(nid)
                    break
        return routes, removed

    def _worst_removal(self, routes):
        """
        Worst Removal: αφαιρεί πελάτες με χειρότερο profit/cost ratio.
        Σε CTOP: αφαιρούμε αυτούς που κοστίζουν πολύ χρόνο σε σχέση
        με το κέρδος τους → ελευθερώνουμε χώρο για καλύτερους.
        """
        m = self.model
        scores = []  # (score, route_idx, node_id)

        for ri, route in enumerate(routes):
            for ni in range(1, len(route) - 1):
                nid = route[ni]
                if nid in self.mandatory_set:
                    continue
                # Κόστος που καταλαμβάνει αυτός ο πελάτης
                prev, succ = route[ni - 1], route[ni + 1]
                cost_contrib = (m.cost_matrix[prev][nid] + m.cost_matrix[nid][succ]
                                - m.cost_matrix[prev][succ])
                profit = m.nodes[nid].profit
                # Χαμηλό score = κακός πελάτης (πολύ κόστος, λίγο κέρδος)
                score = profit / (cost_contrib + 1e-6)
                scores.append((score, ri, nid))

        scores.sort(key=lambda x: x[0])  # χειρότεροι πρώτοι
        removed = []
        for i in range(min(5, len(scores))):
            _, ri, nid = scores[i]
            for r in routes:
                if nid in r:
                    r.remove(nid)
                    removed.append(nid)
                    break
        return routes, removed

    def _shaw_removal(self, routes):
        """
        Shaw Removal: αφαιρεί "παρόμοιους" πελάτες (κοντά + παρόμοιο demand).
        Ελευθερώνει μια ολόκληρη γεωγραφική περιοχή → ο repair μπορεί
        να την ξαναοργανώσει από την αρχή.
        """
        m = self.model
        served = [n for r in routes for n in r
                  if n != 0 and n not in self.mandatory_set]
        if not served:
            return routes, []

        num = max(3, int(len(served) * 0.10))
        # Ξεκίνα από τυχαίο πελάτη
        seed_node = random.choice(served)
        to_remove = [seed_node]

        while len(to_remove) < num:
            ref = random.choice(to_remove)
            # Βρες τον πιο παρόμοιο πελάτη
            candidates = [c for c in served if c not in to_remove]
            if not candidates:
                break
            candidates.sort(key=lambda c: (
                    m.cost_matrix[ref][c] + abs(m.nodes[ref].demand - m.nodes[c].demand)
            ))
            to_remove.append(candidates[0])

        for nid in to_remove:
            for r in routes:
                if nid in r:
                    r.remove(nid)
                    break
        return routes, to_remove

    def _string_removal(self, routes):
        """
        String Removal: αφαιρεί τμήμα συνεχόμενων πελατών από ένα route.
        Πιο aggressive — αναδομεί ολόκληρο κομμάτι route.
        """
        # Βρες routes με τουλάχιστον 3 πελάτες (εκτός depot)
        eligible = [(ri, r) for ri, r in enumerate(routes) if len(r) > 4]
        if not eligible:
            return self._random_removal(routes)

        ri, route = random.choice(eligible)
        # Μέγεθος string: 2 έως μισό route
        max_size = max(2, (len(route) - 2) // 2)
        size = random.randint(2, max_size)
        start = random.randint(1, len(route) - 1 - size)

        removed = []
        indices = list(range(start, start + size))
        # Αφαίρεσε μόνο non-mandatory
        for idx in sorted(indices, reverse=True):
            nid = route[idx]
            if nid != 0 and nid not in self.mandatory_set:
                route.pop(idx)
                removed.append(nid)

        return routes, removed

    # ── REPAIR OPERATORS ─────────────────────────────────────────────

    def _greedy_repair(self, routes, unassigned):
        """
        Greedy Repair: εισάγει κάθε πελάτη στην καλύτερη θέση
        (μέγιστο profit/cost_increase ratio).
        """
        m = self.model
        # Ταξινόμηση κατά κέρδος φθίνουσα
        unassigned.sort(key=lambda nid: m.nodes[nid].profit, reverse=True)

        for nid in unassigned:
            best_score, best_loc = -1, None
            for ri, route in enumerate(routes):
                if route_load(m, route) + m.nodes[nid].demand > m.capacity:
                    continue
                for pos in range(1, len(route)):
                    if not can_insert(m, route, nid, pos):
                        continue
                    delta = insertion_cost_delta(m, route, nid, pos)
                    score = m.nodes[nid].profit / (delta + 1e-6)
                    if score > best_score:
                        best_score = score
                        best_loc = (ri, pos)
            if best_loc:
                routes[best_loc[0]].insert(best_loc[1], nid)

        # Δοκίμασε να βάλεις και ανεξυπηρέτητους πελάτες
        all_served = get_all_served(routes)
        extras = [n.id for n in m.nodes
                  if not n.isDepot and n.id not in all_served]
        extras.sort(key=lambda i: m.nodes[i].profit, reverse=True)

        for nid in extras:
            result = best_insertion(m, routes, nid)
            if result is not None:
                _, r_idx, pos = result
                routes[r_idx].insert(pos, nid)

        return routes

    def _regret_2_repair(self, routes, unassigned):
        """
        Regret-2 Repair: εισάγει πρώτα τον πελάτη με το μεγαλύτερο
        "μετάνιωμα" — δηλαδή αυτόν που αν δεν μπει τώρα, η 2η
        καλύτερη θέση του είναι πολύ χειρότερη.
        """
        m = self.model
        remaining = list(unassigned)

        while remaining:
            regrets = []
            for nid in remaining:
                insertions = []
                for ri, route in enumerate(routes):
                    if route_load(m, route) + m.nodes[nid].demand > m.capacity:
                        continue
                    for pos in range(1, len(route)):
                        if not can_insert(m, route, nid, pos):
                            continue
                        delta = insertion_cost_delta(m, route, nid, pos)
                        # Score: profit - cost (υψηλότερο = καλύτερο)
                        score = m.nodes[nid].profit - delta
                        insertions.append((score, ri, pos))

                if not insertions:
                    regrets.append((-1e9, nid, None))
                    continue

                insertions.sort(key=lambda x: x[0], reverse=True)
                best_score = insertions[0][0]
                second_score = insertions[1][0] if len(insertions) > 1 else best_score - 100
                regret = best_score - second_score
                regrets.append((regret, nid, insertions[0]))

            regrets.sort(key=lambda x: x[0], reverse=True)
            regret_val, nid, best_move = regrets[0]

            if best_move is not None:
                _, ri, pos = best_move
                routes[ri].insert(pos, nid)

            remaining.remove(nid)

        # Δοκίμασε extras
        all_served = get_all_served(routes)
        extras = [n.id for n in m.nodes
                  if not n.isDepot and n.id not in all_served]
        extras.sort(key=lambda i: m.nodes[i].profit, reverse=True)

        for nid in extras:
            result = best_insertion(m, routes, nid)
            if result is not None:
                _, r_idx, pos = result
                routes[r_idx].insert(pos, nid)

        return routes

    # ── ROULETTE WHEEL SELECTION ─────────────────────────────────────

    def _roulette_select(self, weights):
        """
        Roulette Wheel: επιλέγει operator με πιθανότητα ανάλογη του βάρους.
        Operator με βάρος 5.0 έχει 5× πιθανότητα από αυτόν με 1.0.
        """
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    # ── SIMULATED ANNEALING ACCEPTANCE ───────────────────────────────

    def _sa_accept(self, new_profit, new_cost):
        """
        Simulated Annealing: αποδέχεται χειρότερη λύση με πιθανότητα
        P = exp(-Δ / T) όπου Δ = current_profit - new_profit.

        Στην αρχή (T υψηλό) → δέχεται σχεδόν τα πάντα.
        Στο τέλος (T χαμηλό) → δέχεται μόνο βελτιώσεις.
        """
        if new_profit > self.current_profit:
            return True
        if new_profit == self.current_profit and new_cost < self.current_cost:
            return True

        # Δέξου χειρότερη λύση με πιθανότητα exp(-Δ/T)
        delta = self.current_profit - new_profit  # θετικό = χειρότερη
        if self.sa_temperature > 0.01:
            prob = math.exp(-delta / self.sa_temperature)
            return random.random() < prob
        return False

    # ── ΕΝΗΜΕΡΩΣΗ ΒΑΡΩΝ ──────────────────────────────────────────────

    def _update_weights(self):
        """
        Κάθε segment (50 iterations), ενημέρωσε τα βάρη:
        new_weight = decay * old_weight + (1 - decay) * (score / uses)

        Operators με υψηλό score/uses → μεγαλύτερο βάρος →
        μεγαλύτερη πιθανότητα επιλογής.
        """
        for i in range(len(self.destroy_weights)):
            if self.destroy_uses[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_uses[i]
                self.destroy_weights[i] = (self.weight_decay * self.destroy_weights[i]
                                           + (1 - self.weight_decay) * avg_score)
                self.destroy_weights[i] = max(self.destroy_weights[i], 0.1)
            self.destroy_scores[i] = 0
            self.destroy_uses[i] = 0

        for i in range(len(self.repair_weights)):
            if self.repair_uses[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_uses[i]
                self.repair_weights[i] = (self.weight_decay * self.repair_weights[i]
                                          + (1 - self.weight_decay) * avg_score)
                self.repair_weights[i] = max(self.repair_weights[i], 0.1)
            self.repair_scores[i] = 0
            self.repair_uses[i] = 0

    # ── ΚΥΡΙΟΣ ΒΡΟΧΟΣ ALNS ──────────────────────────────────────────

    def run(self, time_limit_sec=180):
        """
        Κύριος βρόχος ALNS.

        Σε κάθε iteration:
          1. Roulette Wheel → διάλεξε destroy + repair operator
          2. Destroy → αφαίρεσε πελάτες από λύση
          3. Repair → ξανάβαλε πελάτες (στοχεύοντας κέρδος)
          4. SA acceptance → αποδέξου ή απόρριψε
          5. Score τους operators ανάλογα με αποτέλεσμα
          6. Κάθε 50 iter → ενημέρωσε βάρη
        """
        t0 = time.time()
        iteration = 0

        print(f"    ALNS starting... (profit={self.best_profit}, cost={self.best_cost:.1f})")

        while time.time() - t0 < time_limit_sec:
            iteration += 1

            # ── 1. Επιλογή operators (Roulette Wheel) ─────────────
            d_idx = self._roulette_select(self.destroy_weights)
            r_idx = self._roulette_select(self.repair_weights)
            d_name, d_func = self.destroy_ops[d_idx]
            r_name, r_func = self.repair_ops[r_idx]

            # ── 2. Αντίγραφο τρέχουσας λύσης ─────────────────────
            new_routes = [r[:] for r in self.routes]

            # ── 3. DESTROY: αφαίρεσε πελάτες ─────────────────────
            new_routes, removed = d_func(new_routes)

            # Καθάρισε κενά routes
            new_routes = [r for r in new_routes if len(r) > 2]
            while len(new_routes) < self.model.vehicles:
                new_routes.append([0, 0])

            # ── 4. REPAIR: ξανάβαλε πελάτες ──────────────────────
            new_routes = r_func(new_routes, removed)

            # Καθάρισε τελική λύση
            new_routes = [r for r in new_routes if len(r) > 2]

            # ── 5. Αξιολόγηση νέας λύσης ─────────────────────────
            new_profit = total_profit(self.model, new_routes)
            new_cost = total_cost(self.model, new_routes)

            # Έλεγχος εγκυρότητας
            valid = True
            for r in new_routes:
                if route_load(self.model, r) > self.model.capacity or route_cost(self.model, r) > self.model.t_max:
                    valid = False
                    break
            if self.enforce_mandatory:
                served = get_all_served(new_routes)
                if not self.mandatory_set.issubset(served):
                    valid = False

            # ── 6. Βαθμολόγηση + αποδοχή ─────────────────────────
            if valid:
                is_new_best = (new_profit > self.best_profit or
                               (new_profit == self.best_profit and new_cost < self.best_cost))
                is_better = (new_profit > self.current_profit or
                             (new_profit == self.current_profit and new_cost < self.current_cost))
                is_accepted = self._sa_accept(new_profit, new_cost)

                if is_new_best:
                    score = self.SCORE_BEST
                    self.best_routes = [r[:] for r in new_routes]
                    self.best_profit = new_profit
                    self.best_cost = new_cost
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_profit
                    self.current_cost = new_cost
                elif is_better:
                    score = self.SCORE_BETTER
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_profit
                    self.current_cost = new_cost
                elif is_accepted:
                    score = self.SCORE_ACCEPTED
                    self.routes = [r[:] for r in new_routes]
                    self.current_profit = new_profit
                    self.current_cost = new_cost
                else:
                    score = self.SCORE_REJECTED
            else:
                score = self.SCORE_REJECTED

            # ── 7. Ενημέρωση scores operators ─────────────────────
            self.destroy_scores[d_idx] += score
            self.destroy_uses[d_idx] += 1
            self.repair_scores[r_idx] += score
            self.repair_uses[r_idx] += 1

            # ── 8. Ενημέρωση βαρών κάθε segment ──────────────────
            if iteration % self.segment_size == 0:
                self._update_weights()

            # ── 9. Cooling θερμοκρασίας SA ────────────────────────
            self.sa_temperature *= self.sa_cooling

            # ── 10. Εκτύπωση ──────────────────────────────────────
            if iteration % 200 == 0:
                elapsed = time.time() - t0
                print(f"    ALNS iter {iteration}: profit={self.current_profit}, "
                      f"best={self.best_profit}, T={self.sa_temperature:.2f}, "
                      f"time={elapsed:.0f}s")
                dw = [f"{n}={w:.2f}" for (n, _), w in zip(self.destroy_ops, self.destroy_weights)]
                rw = [f"{n}={w:.2f}" for (n, _), w in zip(self.repair_ops, self.repair_weights)]
                print(f"      D:[{', '.join(dw)}]")
                print(f"      R:[{', '.join(rw)}]")

        print(f"    ALNS complete: {iteration} iterations, best_profit={self.best_profit}")
        return self.best_routes, self.best_profit, self.best_cost


# ═══════════════════════════════════════════════════════════════════════════
#  ΚΥΡΙΑ ΣΥΝΑΡΤΗΣΗ solve()
# ═══════════════════════════════════════════════════════════════════════════

def solve(model, solution_file, enforce_mandatory=True):
    """
    Λύνει το CTOP σε 3 φάσεις:
      Φάση 1: Adaptive Greedy Construction (~instant)
      Φάση 2: Tabu Search (~80 δευτερόλεπτα)
      Φάση 3: ALNS (~180 δευτερόλεπτα)
      Σύνολο: ~260 δευτ. < 5 λεπτά
    """
    random.seed(seed)

    # ── Φάση 1 ────────────────────────────────────────────────────────
    print("  Phase 1: Adaptive Greedy Construction...")
    routes = adaptive_greedy_construct(model, enforce_mandatory)
    p1 = total_profit(model, routes)
    c1 = total_cost(model, routes)
    print(f"  → profit={p1}, cost={c1:.1f}, routes={len(routes)}")

    # ── Φάση 2 ────────────────────────────────────────────────────────
    print("  Phase 2: Tabu Search...")
    ts = TabuSearchCTOP(model, routes, enforce_mandatory)
    routes, p2, c2 = ts.run(max_iter=2000, time_limit_sec=80)
    routes = [r for r in routes if len(r) > 2]
    print(f"  → profit={p2}, cost={c2:.1f}")

    # ── Φάση 3 ────────────────────────────────────────────────────────
    print("  Phase 3: ALNS...")
    alns = ALNS_CTOP(model, routes, enforce_mandatory)
    routes, p3, c3 = alns.run(time_limit_sec=180)
    routes = [r for r in routes if len(r) > 2]
    print(f"  → profit={p3}, cost={c3:.1f}")

    # ── Αποτελέσματα ──────────────────────────────────────────────────
    print(f"\n  === SUMMARY ===")
    print(f"  Greedy:     profit={p1}")
    print(f"  +Tabu:      profit={p2} (+{p2 - p1})")
    print(f"  +ALNS:      profit={p3} (+{p3 - p2})")
    print(f"  Total gain: +{p3 - p1} profit")

    write_solution(routes, solution_file)
    print(f"  Solution written to {solution_file}")