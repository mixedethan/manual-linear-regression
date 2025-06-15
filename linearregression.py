import numpy as np
import matplotlib.pyplot as plt

# minimize sum of squared errors / vertical distance to find line of best fit
# SSE = ∑ᵢⁿ (yᵢ - yhatᵢ )²

def compute_error(m, b, points):
    # Error(m, b) = 1/n ∑ᵢⁿ (yᵢ - (mxᵢ + b))²
    totalError = 0
    N = float(len(points))
    # for every point
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (m * x + b))**2

    return totalError / N

def gradient_descent(points, initial_m, initial_b, learning_rate, num_iters):
    m = initial_m
    b = initial_b
    
    for i in range(num_iters):
        # update b and m w/ new and more accurate b and m by performing a gradient step
        m, b = step_gradient(m, b, np.array(points), learning_rate)

    return [m, b]

def step_gradient(m, b, points, learning_rate):

    # starting points for gradient
    m_gradient = 0
    b_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # computing partial derivatives of the error function to find direction for gradient
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += -(2/N) * x * (y - ((m * x) + b))

    # update b and m using partial derivatives
    new_m = m - (learning_rate * m_gradient)
    new_b = b - (learning_rate * b_gradient)
    
    return [new_m, new_b]


def plot_regression(points, m, b):
    x = points[:, 0]
    y = points[:, 1]

    ypred = m * x + b
    equation = f'y = {m:.2f}x + {b:.2f}'
    x_pos = min(x) + 1
    y_pos = max(y) - 5

    plt.scatter(x, y, color='blue',s=40, edgecolors='black', alpha=0.8, label='Actual Data')
    plt.plot(x, ypred, color='red',linewidth=2.5, label='Line of Best Fit')
    plt.xlabel('Hours Spent Studying by Student', fontsize=12)
    plt.ylabel('Test Scores of Students', fontsize=12)
    plt.title('Regression: Hours Studied vs. Test Scores w/ Regression', fontsize=14, weight='bold')
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', linestyle=':', color='lightgray', linewidth=0.4)
    plt.text(x_pos, y_pos, equation, fontsize=10, color='red', weight='bold',
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.show()

def loaddata(filepath):
    return np.genfromtxt(filepath, delimiter=',')

def main():

    # data in format, [hour studied, test scores], 2D array
    points = loaddata('data.csv')

    # how much we should adjust the model based off the errors we perceive
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iters = 1000

    # train the model
    print(f'Starting gradient descent:\nm = {initial_m}\nb = {initial_b}\nerror = {compute_error(initial_m, initial_b, points)}\n')
    [m, b] = gradient_descent(points, initial_m, initial_b, learning_rate, num_iters)
    print(f'After {num_iters} iterations...\nEnding gradient descent:\nm = {m}\nb = {b}\nerror = {compute_error(m, b, points)}')

    # visualizations
    plot_regression(points, m, b)

if __name__ == "__main__":
    main()
