import numpy as np

def knapsack_to_qubo(valeurs, poids, poids_max, penalty_factor=None):
    n_items = len(valeurs)
    num_slack_bits = int(np.floor(np.log2(poids_max))) + 1
    
    qubo_size = n_items + num_slack_bits
    Q = np.zeros((qubo_size, qubo_size))
    
    if penalty_factor is None:
        penalty_factor = max(valeurs) + 1
        
    #minimiser -V
    for i in range(n_items):
        Q[i, i] -= valeurs[i]
        
    # pénalités objets
    for i in range(n_items):
        Q[i, i] += penalty_factor * (poids[i]**2 - 2 * poids_max * poids[i])
        for j in range(i + 1, n_items):
            Q[i, j] += 2 * penalty_factor * poids[i] * poids[j]
            
    # pénalités slack
    for k in range(num_slack_bits):
        slack_w = 2**k
        idx = n_items + k
        Q[idx, idx] += penalty_factor * (slack_w**2 - 2 * poids_max * slack_w)
        
        for k2 in range(k + 1, num_slack_bits):
            Q[idx, n_items + k2] += 2 * penalty_factor * slack_w * (2**k2)
            
    for i in range(n_items):
        for k in range(num_slack_bits):
            Q[i, n_items + k] += 2 * penalty_factor * poids[i] * (2**k)

    return Q