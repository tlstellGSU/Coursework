import numpy as np
import matplotlib.pyplot as plt

# Pitchfork bifurcation: dx/dt = r*x - x**3
# We'll shade the (r,x) plane where dx/dt > 0 (increasing) in light green
# and where dx/dt < 0 (decreasing) in light red. We'll overlay equilibria.

r_values = np.linspace(-2, 2, 400)
x_values = np.linspace(-2, 2, 400)

# Create meshgrid with R along x-axis and X along y-axis for plotting
R, X = np.meshgrid(r_values, x_values)
DX = R * X - X**3

plt.figure(figsize=(10, 6))
# Shade regions: DX < 0 (decreasing) red, DX > 0 (increasing) green
# Use contourf with two regions split at DX=0
# levels chosen to partition below/above zero
plt.contourf(
    R,
    X,
    DX,
    levels=[DX.min() - 1, 0, DX.max() + 1],
    colors=["lightcoral", "lightgreen"],
    alpha=0.5,
)
# Draw the nullcline DX=0 for clarity
plt.contour(R, X, DX, levels=[0], colors="k", linewidths=0.6)

# Plot stable equilibria (solid blue) and unstable (dashed red)
r_neg = r_values[r_values < 0]
r_pos = r_values[r_values > 0]
plt.plot(r_neg, np.zeros_like(r_neg), "b-", label="Stable Equilibria (x=0 for r<0)")
plt.plot(r_pos, np.sqrt(r_pos), "b-")
plt.plot(r_pos, -np.sqrt(r_pos), "b-")
# Unstable equilibrium
# only at x=0 for r>0

plt.plot(r_pos, np.zeros_like(r_pos), "r--", label="Unstable Equilibrium (x=0)")
#plt.plot(r_values, np.zeros_like(r_values), "r--", label="Unstable Equilibrium (x=0)")

plt.title("Pitchfork Bifurcation Diagram (shaded by sign of $dx/dt$)")
plt.xlabel("Parameter r")
plt.ylabel("State x")
plt.ylim(-2, 2)
plt.xlim(-2, 2)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig("pitchfork_bifurcation.png", dpi=300)