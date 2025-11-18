import numpy as np
import matplotlib.pyplot as plt

base_function = lambda x: -(np.exp(1 / (0.4 * (np.sin(x) + 2))) - 0.01 * x**2)
# bound from -20 to 20

# simulated annealing


def simulated_annealing(
    func,
    x_0,
    upper_bound,
    lower_bound,
    initial_temp=1000,
    cooling_rate=0.99,
    max_iter=10000,
):
    current_x = x_0
    current_y = func(current_x)
    best_x = current_x
    best_y = current_y
    temp = initial_temp

    guesses = [best_x]

    for i in range(max_iter):
        new_x = current_x + np.random.uniform(-1, 1)
        new_x = np.clip(new_x, lower_bound, upper_bound)
        new_y = func(new_x)

        delta_y = new_y - current_y

        if delta_y < 0 or np.random.rand() < np.exp(-delta_y / temp):
            current_x = new_x
            current_y = new_y

            if current_y < best_y:
                best_x = current_x
                best_y = current_y

        temp *= cooling_rate

        guesses.append(best_x)

    return best_x, best_y, guesses


# plot results and initial guess and first 5 guesses

best_x, best_y, guesses = simulated_annealing(
    base_function, np.random.uniform(-20, 20), 20, -20
)

x = np.linspace(-20, 20, 1000)
y = base_function(x)

initial_x = np.random.uniform(-20, 20)
initial_y = base_function(initial_x)

plt.plot(x, y, label="Base Function")
plt.scatter(best_x, best_y, color="red", label="Best Solution")
plt.scatter(initial_x, initial_y, color="green", label="Initial Guess")

# Plot the first 5 guesses
for i, guess in enumerate(guesses[:5]):
    plt.scatter(guess, base_function(guess), color="blue", label=f"Guess {i+1}")

plt.legend()
plt.show()
