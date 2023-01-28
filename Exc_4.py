import numpy as np
import matplotlib.pyplot as plt
import math


#Config


max_time = 20
time_steps = 10
max_iterations = 2

start_capital = 100
precision = 1e-10

def g(x):
    return 2*x**0.5

def g_derivative(x):
    return 0.5*x**-0.5

def f(x, f_number):
    if f_number == 1:
        return x
    elif f_number == 2:
        return x + x**2/10
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
    X_old = np.zeros((time_steps+1)) + start_capital
    X_new = X_old
    L = np.zeros((time_steps)) 
    
    for i in range(max_iterations):
        L[time_steps-1] = g_derivative(start_capital)
        np.linalg.norm(X_new-X_old)
        for j in range(time_steps-2, -1, -1):
            L[j] = L[j+1] + dt*f(X_old[j],f_number)*L[j+1]

        for k in range(time_steps):
            X_new[k+1] = (X_new[k]+dt*(f(X_new[k],f_number)-1/(L[k]**(3/5))))

        X_old = X_new
        if np.linalg.norm(X_new-X_old) < precision:
            break
    X = X_new.T
    alpha = (L**(-3/5)).T
    return X, alpha



def plot_capital(x):
    time_axis = np.arange(0, time_steps+1)
    print(time_axis.shape, x.shape)
    plt.plot(time_axis, x, 'o')
    plt.show()
    plt.ylim(0, 1000)

def plot_alpha(x):
    time_axis = np.arange(1, time_steps+1, 1)
    print(time_axis.shape, x.shape)
    plt.plot(time_axis, x, 'o')
    plt.show()
    
    


capital, alpha = EulerLagrange(time_steps, max_iterations, start_capital ,precision,1)

plot_capital(capital)
plot_alpha(alpha)

