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
        return x + 0.1*x**2
    
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
        L[time_steps-1] = g_derivative(start_capital)
        for j in range(time_steps-2, -1, -1):
            L[j] = L[j+1] + dt*f_derivative(X_old[j],f_number)*L[j+1]
        for k in range(time_steps):
            X_new[k+1] = X_new[k]+dt*(f(X_old[k],f_number)-1/(L[k-1]**(3/5)))
        X_old = X_new
        if numpy.linalg.norm(X_new-X_old) < precision:
            print(i)
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
    


capital, alpha = EulerLagrange(time_steps, max_iterations, start_capital ,precision,2)
#plot_capital(capital)
#plot_alpha(alpha)

#Plot capital and spending over f_number 1 to 3 in one plot
for i in range(1,4):
    capital, alpha = EulerLagrange(time_steps, max_iterations, start_capital ,precision,i)
    time_axis = numpy.arange(0, time_steps+1)
    plt.plot(time_axis, capital, label='Data', marker='o')
    time_axis = numpy.arange(1, time_steps+1, 1)
    plt.plot(time_axis, alpha, label='Data', marker='o')
    plt. title('Capital and spending over time for f_number = ' + str(i))
    plt.show()






