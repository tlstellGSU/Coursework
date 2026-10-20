import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from pathlib import Path

# Resolve the absolute path of the directory containing the script
file_dir = Path(__file__).resolve().parent
save_path = file_dir / "sde_pdf_plot.png"

def plot_sde_quasistationary_pdf(N=100, save_path=save_path):
    # Define the continuous state space interval
    x = np.linspace(1, N, 1000)
    
    # Compute log-density for numerical stability:
    # \ln(p(x)) \propto x + 99 \ln(150 - x) - \ln(x)
    log_p = x + 99 * np.log(150 - x) - np.log(x)
    
    # Shift by the maximum log-density to prevent overflow upon exponentiation
    log_p -= np.max(log_p)
    p_unnormalized = np.exp(log_p)
    
    # Compute the normalization constant C via Simpson's rule
    C_inv = simpson(p_unnormalized, x=x)
    p_normalized = p_unnormalized / C_inv
    
    # Generate the figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, p_normalized, label=r'SDE Approximation $\hat{p}(x)$', color='blue', linewidth=2)
    
    # Formatting the plot
    ax.set_title("Quasistationary PDF of the SDE SIS Model", fontsize=14)
    ax.set_xlabel(r"Number of Infected Individuals ($x$)", fontsize=12)
    ax.set_ylabel(r"Probability Density $\hat{p}(x)$", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    # Save to disk (matplotlib accepts pathlib.Path objects natively)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return x, p_normalized

if __name__ == "__main__":
    x, p = plot_sde_quasistationary_pdf()
    print(f"Plot saved to: {save_path}")