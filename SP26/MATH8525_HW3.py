import numpy as np 
import matplotlib.pyplot as plt
import os

# Problem 3.17.b

def compute_q_star(N, b, d):

    q_star = np.zeros(N)
    
    q_star[0] = 1

    for i in range(N-1):
        frac_i = b[i]/d[i+1]
        q_star[i+1] = frac_i * q_star[i]
        
    q_star_norm = np.linalg.norm(q_star, ord=1)
    q_star = q_star/q_star_norm

    return q_star

def problem_3_17_d_list(N, r):
    d = np.zeros(N)
    for i in range(N):
        state = i+1
        d[i] = r*state**2/N

    return d

def problem_3_17_b_list(N, r):
    b = np.zeros(N)
    for i in range(N):
        state = i+1
        b[i] = r*state*(N-state)/N

    return b

def plot_q_star(q_star, save_path=None):
    # Use a specific figure size for consistency
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1-based indexing for the x-axis to match population states
    ax.plot(range(1, len(q_star) + 1), q_star, marker='o', linestyle='-', markersize=4)
    
    ax.set_xlabel("State (i)")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    
    if save_path is not None:
        # bbox_inches='tight' prevents labels from being cut off
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    # Crucial: Close the figure to free memory and reset for the next plot
    plt.close(fig)

def problem_3_17_main(N=50, r=0.01, save_path=None):
    q_star = compute_q_star(N, problem_3_17_b_list(N, r), problem_3_17_d_list(N, r))
    if save_path is not None:
        full_path = os.path.join(save_path, "problem_3_17.png")
        plot_q_star(q_star, full_path)
    else:
        plot_q_star(q_star)

# problem 3.20

def problem_3_20_b_list(N, beta):
    b = np.zeros(N)
    for i in range(N):
        state = i+1
        b[i] = beta*state * (N-state)
    return b

def problem_3_20_d_list(N, gamma):
    d = np.zeros(N)
    for i in range(N):
        state = i+1
        d[i] = gamma*state
    return d

def problem_3_20_main(N=20, beta=0.01, gamma=0.0025, save_path=None):
    q_star = compute_q_star(N, problem_3_20_b_list(N, beta), problem_3_20_d_list(N, gamma))
    if save_path is not None:
        full_path = os.path.join(save_path, "problem_3_20.png")
        plot_q_star(q_star, full_path)
    else:
        plot_q_star(q_star)

def calculate_duration_probs(p):
    """
    Calculates P(T=n) for n=1, 2, 3 given probability p.
    q = 1 - p
    Derived formulas:
    P(T=1) = q^2
    P(T=2) = 2*p*q^2 + p^2
    P(T=3) = 2*p^2*q
    """
    q = 1 - p
    p1 = q**2
    p2 = 2 * p * q**2 + p**2
    p3 = 2 * p**2 * q
    
    return [p1, p2, p3]

def calculate_expected_val(probs):
    """Calculates E[T] = sum(t * P(T=t))"""
    e_t = 0
    for t_idx, prob in enumerate(probs):
        duration = t_idx + 1 # 0-index corresponds to T=1
        e_t += duration * prob
    return e_t

def plot_duration_distribution(p_values, save_path=None):
    """
    Plots a grouped bar chart for different p values.
    """
    # Setup data
    durations = [1, 2, 3] # Possible duration times
    x = np.arange(len(durations)) # Label locations
    width = 0.25 # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # Loop through p values and plot offset bars
    for i, p in enumerate(p_values):
        probs = calculate_duration_probs(p)
        e_t = calculate_expected_val(probs)
        
        # Offset the x position for each p value
        offset = (i - 1) * width 
        rects = ax.bar(x + offset, probs, width, label=f'p={p} (E[T]={e_t:.2f})')
        
        # Optional: Add value labels on top of bars
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    # Formatting
    ax.set_xlabel('Duration of Epidemic (T)')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution of Epidemic Duration (Chain Binomial)')
    ax.set_xticks(x)
    ax.set_xticklabels(durations)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Save or Show
    if save_path:
        full_path = os.path.join(save_path, "problem_3_22_duration.png")
        fig.savefig(full_path, bbox_inches='tight')
        print(f"Saved: {full_path}")
    else:
        plt.show()
    
    plt.close(fig)

def calculate_magnitude_probs(p, W=3):
    
    q = 1 - p
    p0 = q**2
    p1 = 2*p*q**2
    p2 = p**2 + 2*p**2*q
    
    return [p0, p1, p2]

def expected_value_epi_magnitude(p, W=3):
    
    expected_value = 0

    q = 1 - p
    probs = calculate_magnitude_probs(p, W)
    
    for W in range(W):
        expected_value += W * probs[W]
        
    return expected_value

def plot_magnitude_distribution(p_values, save_path=None):
    """
    Plots a grouped bar chart for different p values.
    """
    # Setup data
    magnitudes = [0, 1, 2] # Possible magnitude values
    x = np.arange(len(magnitudes)) # Label locations
    width = 0.25 # Width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # Loop through p values and plot offset bars
    for i, p in enumerate(p_values):
        probs = calculate_magnitude_probs(p)
        e_w = expected_value_epi_magnitude(p)
        
        # Offset the x position for each p value
        offset = (i - 1) * width 
        rects = ax.bar(x + offset, probs, width, label=f'p={p} (E[W]={e_w:.2f})')
        
        # Optional: Add value labels on top of bars
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    # Formatting
    ax.set_xlabel('Magnitude of Epidemic (W)')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution of Epidemic Magnitude (Chain Binomial)')
    ax.set_xticks(x)
    ax.set_xticklabels(magnitudes)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Save or Show
    if save_path:
        full_path = os.path.join(save_path, "problem_3_22_magnitude.png")
        fig.savefig(full_path, bbox_inches='tight')
        print(f"Saved: {full_path}")
    else:
        plt.show()
    
    plt.close(fig)

def problem_3_22_main(save_path=None):
    p_values = [0.2, 0.5, 0.8]
    plot_duration_distribution(p_values, save_path)
    plot_magnitude_distribution(p_values, save_path)

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "plots", "HW3")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    problem_3_17_main(N=50, r=0.01, save_path=save_dir)
    problem_3_20_main(N=20, beta=0.01, gamma=0.0025, save_path=save_dir)
    problem_3_22_main(save_path=save_dir)