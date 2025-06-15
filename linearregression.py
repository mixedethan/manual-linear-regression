import numpy as np

# minimize sum of squared errors / vertical distance to find line of best fit
# SSE = ∑ᵢⁿ (yᵢ - yhatᵢ )²

def compute_error(b, m, points):
    # Error(m, b) = 1/n ∑ᵢⁿ (yᵢ - (mxᵢ + b))²
    totalError = 0
    N = float(len(points))
    # for every point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (m * x + b))**2

    return totalError / N

def gradient_descent(points, initial_b, initial_m, learning_rate, num_iters):
    b = initial_b
    m = initial_m

    for i in range(num_iters):
        # update b and m w/ new and more accurate b and m by performing a gradient step
        b, m = step_gradient(b, m, np.array(points), learning_rate)

    return [b, m]

def step_gradient(b, m, points, learning_rate):

    # starting points for gradient
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # computing partial derivatives of the error function to find direction for gradient
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += -(2/N) * x * (y - ((m * x) + b))

    # update b and m using partial derivatives
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)

    return [new_b, new_m]

def main():

    # data in format, [hour studied, test scores], 2D array
    points = np.genfromtxt('data.csv', delimiter=',')

    # how much we should adjust the model based off the errors we perceive
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iters = 1000

    # train the model
    print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, points)}\n')
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iters)
    print(f'After {num_iters} iterations, ending gradient descent at b = {b}, m = {m}, error = {compute_error(b, m, points)}')


if __name__ == "__main__":
    main()
