


# Compute competence as the average of the results in the given window
def compute_competence(results_window):
    return sum(results_window)/ len(results_window)

# Compute learning progress as:
#   difference between the competence of the last l results and the competence of the previous l results
def compute_lp(results, n_eval, l=10):
    c_recent = compute_competence(results[n_eval-l+1 : n_eval+1])
    c_older  = compute_competence(results[n_eval-2*l+1 : n_eval-l+1])
    return c_recent - c_older 