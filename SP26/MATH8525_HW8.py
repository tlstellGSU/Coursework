import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_bridge(a, b, num_steps, num_paths):
    """
    Simulates sample paths of a Brownian bridge from a to b using Euler-Maruyama.
    """
    dt = 1.0 / num_steps
    t = np.linspace(0, 1, num_steps + 1)
    X = np.zeros((num_paths, num_steps + 1))
    X[:, 0] = a
    
    # Iterate up to the second-to-last step to avoid the singularity at t=1
    for i in range(num_steps - 1):
        Z = np.random.normal(0, 1, num_paths)
        X[:, i+1] = X[:, i] + ((b - X[:, i]) / (1 - t[i])) * dt + np.sqrt(dt) * Z
        
    # Enforce the terminal condition
    X[:, -1] = b
    
    return t, X

# Generate and plot 3 sample paths
t, paths = simulate_brownian_bridge(a=0.0, b=1.0, num_steps=1000, num_paths=3)

for i in range(3):
    plt.plot(t, paths[i], label=f"Path {i+1}")

plt.xlabel("t")
plt.ylabel("X(t)")
plt.title("Brownian Bridge Sample Paths")
plt.legend()
plt.savefig("brownian_bridge_paths.png")  # Save the figure as a PNG file
plt.show()