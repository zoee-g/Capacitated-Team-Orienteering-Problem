import Parser
import time

def build_greedy_solution(N, K, Q, T_max, profits, demands, mandatory, time_matrix, use_mandatory_logic=False):

    routes = [[0, 0] for _ in range(K)]
    current_loads = [0] * K
    current_times = [0] * K

    available_customers = list(range(1, N + 1))
    
    # Επειδή παρατηρήσαμε ότι γεμίζει πρώτα η χωρητικότητα, αυξάνουμε την σταθερη ποινή της ωστε να δινεται μεγαλυτερη έμφαση στον χρόνο.
    TIME_WEIGHT = 1.0
    CAPACITY_WEIGHT = 4.0

    while available_customers:
        best_score = -1
        best_insertion = None

        for customer in available_customers:
            actual_profit = profits[customer]

            # Αυξάνουμε υπερβολικά το κέρδος για υποχρεωτικούς πελάτες ώστε να εξασφαλίσουμε την επιλογή τους
            if use_mandatory_logic and mandatory[customer] == 1:
                actual_profit += 10**9 

            # Υπολογίζουμε ποσοστιαία διαθεσιμότητα χρόνου και χωρητικότητας για κάθε όχημα
            for v_idx in range(K):
                time_left_pct = (T_max - current_times[v_idx]) / T_max if T_max > 0 else 0
                cap_left_pct = (Q - current_loads[v_idx]) / Q if Q > 0 else 0
                
                # Υπολογίζουμε τις μεταβλητές α και β για την ποινή, συνυπολογίζοντας τα σταθερά βάρη και την τρέχουσα διαθεσιμότητα
                alpha = TIME_WEIGHT / (time_left_pct + 0.01)
                beta = CAPACITY_WEIGHT / (cap_left_pct + 0.01)

                # Εξετάζουμε όλες τις πιθανές θέσεις εισαγωγής του πελάτη στο δρομολόγιο του οχήματος
                for i in range(1, len(routes[v_idx])):
                    prev_node = routes[v_idx][i-1]
                    next_node = routes[v_idx][i]
                    
                    # Έλεγχος Χωρητικότητας
                    new_load = current_loads[v_idx] + demands[customer]
                    if new_load > Q:
                        continue
                    
                    # Έλεγχος Χρόνου
                    added_dist = time_matrix[prev_node][customer] + time_matrix[customer][next_node]
                    removed_dist = time_matrix[prev_node][next_node]
                    new_time = current_times[v_idx] + added_dist - removed_dist
                    
                    if new_time <= T_max:
                        # Υπολογισμός Σκορ με κσνονικοποίηση κόστους και κέρδους
                        cost_increase = added_dist - removed_dist
                        
                        # Μετατρέπουμε τα κόστη σε ποσοστά της συνολικής διαθεσιμότητας
                        normalized_time_cost = cost_increase / T_max
                        # Μετατρέπουμε τη ζήτηση σε ποσοστό της χωρητικότητας
                        normalized_demand = demands[customer] / Q
                        
                        # Υπολογίζουμε την ποινή και το σκορ
                        penalty = (alpha * normalized_time_cost) + (beta * normalized_demand)          
                        score = actual_profit / penalty
                        
                        # Ενημερώνουμε την καλύτερη εισαγωγή αν το σκορ είναι υψηλότερο
                        if score > best_score:
                            best_score = score
                            best_insertion = (customer, v_idx, i, new_load, new_time)
        
        # Αν βρέθηκε έγκυρη εισαγωγή, την εκτελούμε
        if best_insertion:
            cust, v_idx, pos, n_load, n_time = best_insertion
            routes[v_idx].insert(pos, cust)
            current_loads[v_idx] = n_load
            current_times[v_idx] = n_time
            available_customers.remove(cust)
        else:
            break
            
    return routes

if __name__ == "__main__":
    instance_name = "ctop_main_instance.txt"
    print(f"Loading {instance_name}...")
    
    # Φόρτωση του μοντέλου
    model = Parser.load_model(instance_name)
    
    # Εξαγωγή παραμέτρων
    N = model.num_nodes - 1 
    K = model.vehicles
    Q = model.capacity
    T_max = model.t_max
    time_matrix = model.cost_matrix
    profits = [node.profit for node in model.nodes]
    demands = [node.demand for node in model.nodes]
    mandatory = [1 if node.isMandatory else 0 for node in model.nodes]

    # Επίλυση πρώτου σεναρίου
    print("Solving Scenario 1 (Standard)...")
    start_time_1 = time.time()
    sol1 = build_greedy_solution(N, K, Q, T_max, profits, demands, mandatory, time_matrix, use_mandatory_logic=False)
    end_time_1 = time.time()

    # Εγγραφή στο αρχείο
    with open("solution_no_mandatory.txt", 'w', encoding='utf-8') as f:
        for route in sol1:
            f.write(" ".join(map(str, route)) + "\n")
    print("Solution saved to solution_no_mandatory.txt.")
    print(f" Χρόνος εκτέλεσης Σεναρίου 1: {end_time_1 - start_time_1:.2f} δευτερόλεπτα")

    # Επίλυση δευτερου σεναρίου
    print("Solving Scenario 2 (Mandatory)...")
    start_time_2 = time.time()
    sol2 = build_greedy_solution(N, K, Q, T_max, profits, demands, mandatory, time_matrix, use_mandatory_logic=True)
    end_time_2 = time.time()

    # Εγγραφή στο αρχείο
    with open("solution_mandatory.txt", 'w', encoding='utf-8') as f:
        for route in sol2:
            f.write(" ".join(map(str, route)) + "\n")
    print("Solution saved to solution_mandatory.txt")
    print(f"-> Χρόνος εκτέλεσης Σεναρίου 2: {end_time_2 - start_time_2:.2f} δευτερόλεπτα")
    
    print("\nReady to run Main.py")
