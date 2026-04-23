
"""
This module implements the learning progress (LP) computation for the goals in the dataset.
The learning progress for each goal is computed based on the results of the agent's attempts to achieve that goal over time.
"""

# Compute competence as the average of the results in the given window
def compute_competence(results_window):
    return sum(results_window)/ len(results_window)

# Compute learning progress as:
#   difference between the competence of the last l results and the competence of the previous l results
def compute_lp(results, n_eval, l=10):
    # Not enough evaluations yet to fill both windows → return 0
    if n_eval < 2 * l :
        return 0.0
    
    c_recent = compute_competence(results[n_eval-l : n_eval])  # Last l results
    c_older  = compute_competence(results[n_eval-2*l : n_eval-l]) # Previous l results

    return c_recent - c_older 