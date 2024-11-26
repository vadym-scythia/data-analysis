import numpy as np
import matplotlib.pyplot as plt

# Data from the table
# Initialization of arrays x and y representing the discrete points given in the table.
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1.1, 5.9, 13.2, 21.8, 33.4, 45.4])

# Implementation of the least squares method for linear approximation
# The least squares method is used to find the coefficients of the linear polynomial F(x) = a0 + a1 * x,
# which minimize the sum of squared deviations between the function values and the given points.
# Formulas for the coefficients:
# a1 = (n * Σ(x*y) - Σ(x) * Σ(y)) / (n * Σ(x^2) - (Σ(x))^2)
# a0 = (Σ(y) - a1 * Σ(x)) / n
def least_squares_linear(x, y):
    n = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x**2)
    Sxy = np.sum(x * y)
    
    a1 = (n * Sxy - Sx * Sy) / (n * Sxx - Sx**2)
    a0 = (Sy - a1 * Sx) / n
    
    return a0, a1

# Implementation of the least squares method for quadratic approximation
# The least squares method is used to find the coefficients of the quadratic polynomial F(x) = a0 + a1 * x + a2 * x^2,
# which minimize the sum of squared deviations between the function values and the given points.
# The formulas for the coefficients are determined from the system of equations:
# Σ(y) = a0 * n + a1 * Σ(x) + a2 * Σ(x^2)
# Σ(x*y) = a0 * Σ(x) + a1 * Σ(x^2) + a2 * Σ(x^3)
# Σ(x^2*y) = a0 * Σ(x^2) + a1 * Σ(x^3) + a2 * Σ(x^4)
def least_squares_quadratic(x, y):
    n = len(x)
    Sx = np.sum(x)
    Sxx = np.sum(x**2)
    Sxxx = np.sum(x**3)
    Sxxxx = np.sum(x**4)
    Sy = np.sum(y)
    Sxy = np.sum(x * y)
    Sxxy = np.sum(x**2 * y)
    
    # Constructing the system of equations
    A = np.array([
        [n, Sx, Sxx],
        [Sx, Sxx, Sxxx],
        [Sxx, Sxxx, Sxxxx]
    ])
    B = np.array([Sy, Sxy, Sxxy])
    
    # Solving the linear system to find the coefficients
    a0, a1, a2 = np.linalg.solve(A, B)
    
    return a0, a1, a2

# Calculation of coefficients for linear and quadratic approximations
# Using the previously defined functions to find the polynomial coefficients.
a0_linear, a1_linear = least_squares_linear(x, y)
a0_quad, a1_quad, a2_quad = least_squares_quadratic(x, y)

# Creating functions for approximations
# Linear approximation: F1(x) = a0 + a1 * x
def linear_approx(x):
    return a0_linear + a1_linear * x

# Quadratic approximation: F2(x) = a0 + a1 * x + a2 * x^2
def quadratic_approx(x):
    return a0_quad + a1_quad * x + a2_quad * x**2

# Generating points for plotting
# Generating 500 points between the minimum and maximum values of x to create smooth approximation plots.
x_line = np.linspace(min(x), max(x), 500)

y_linear = linear_approx(x_line)
y_quadratic = quadratic_approx(x_line)

# Determining the maximum absolute error for linear and quadratic approximations
# The error is defined as the maximum absolute difference between the actual values and the approximated values at points x.
# Formula for error: max_error = max(|y_i - F(x_i)|)
y_linear_approx = linear_approx(x)
y_quadratic_approx = quadratic_approx(x)

max_error_linear = np.max(np.abs(y - y_linear_approx))
max_error_quadratic = np.max(np.abs(y - y_quadratic_approx))

print(f"Maximum absolute error for linear approximation: {max_error_linear}")
print(f"Maximum absolute error for quadratic approximation: {max_error_quadratic}")

# Plotting graphs
# Plotting the graph with discrete points and approximation functions.
plt.scatter(x, y, color='red', label='Discrete points')
plt.plot(x_line, y_linear, label='Linear approximation (LSM)', linestyle='--')
plt.plot(x_line, y_quadratic, label='Quadratic approximation (LSM)', linestyle='-.')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Least Squares Approximation')
plt.grid(True)
plt.show()
