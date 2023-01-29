import numpy as numpy
import matplotlib.pyplot as plt

#Config


time_steps = 60
max_iterations = 10
start_capital = 1
precision = 1e-6

def g(x):
    return 2*x**0.5

def g_derivative(x):
    return x**-0.5

def f(x, f_number):
    if f_number == 1:
        return x
    elif f_number == 2:
        return x + (x**2)/10
    elif f_number == 3:
        a = 1.0708944
        b = 0.06184937
        return a*x+b*x**2

def f_derivative(x, f_number):
    if f_number == 1:
        return 1
    elif f_number == 2:
        return 1 + x/5
    elif f_number == 3:
        a = 1.0708944
        b = 0.06184937
        return a + 2*b*x

def EulerLagrange(time_steps, max_iterations, start_capital ,precision, f_number):
    dt = 1/time_steps
    X_old = numpy.zeros((time_steps+1)) + start_capital
    X_new = numpy.zeros((time_steps+1))
    X_new[0] = start_capital
    L = numpy.zeros((time_steps)) 

    for i in range(max_iterations):
        print(i)
        L[time_steps-1] = g_derivative(start_capital)
        for j in range(time_steps-2, -1, -1):
            L[j] = L[j+1] + dt*f_derivative(X_old[j],f_number)*L[j+1]
        for k in range(time_steps):
            X_new[k+1] = (X_new[k]+dt*(f(X_old[k],f_number)-1/(L[k]**(3/5))))
        X_old = X_new
        print(X_new)
        if numpy.linalg.norm(X_new-X_old) < precision:
            break
    X = X_new.T
    alpha = (L**(-3/5)).T
    return X, alpha


def plot_capital(x):
    time_axis = numpy.arange(0, time_steps+1)
    plt.plot(time_axis, x, label='Data', marker='o')
    plt. title('Capital over time')
    plt.show()
    
def plot_alpha(x):
    time_axis = numpy.arange(1, time_steps+1, 1)
    plt.plot(time_axis, x, label='Data', marker='o')
    plt. title('Spending over time')
    plt.show()
    
def plota(x,a):
    time_axis = numpy.arange(1, time_steps+1, 1)
    fig, ax = plt.subplots()
    ax.plot(time_axis, x, label='Capital')
    ax.plot(time_axis, a, label='Spending')
    ax.legend()
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    plt.show()

capital, alpha = EulerLagrange(time_steps, max_iterations, start_capital ,precision,2)
#plot_capital(capital)
#plot_alpha(alpha)


