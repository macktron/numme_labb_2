import numpy as np
import matplotlib.pyplot as plt
import math

#make vector that 

def f(x):
    return -0.5*x**-(3/5)

def f_derivative(x):
    return (3/10)*x**(-8/5)


def lambda_n_i_iteration(lambda_n1_i, x_n_i, dt):
    return lambda_n1_i + dt*f_derivative(x_n_i)*lambda_n1_i
    
def x1_i1_iteration_time(lambda_n1_i, x_n_i1, dt):
    return x_n_i1 + dt*(f(x_n_i1)-(lambda_n1_i**-(3/5)))
    


def iteration_method(x0, lambda0, dt, iterations, N):
    time = np.linspace(0, N, iterations)
    x = np.zeros(iterations)
    lamb = np.zeros(iterations)
    x[0] = x0
    lamb[0] = lambda0
    for j in range(0, N):

        for i in range(0, iterations-1):
            lamb[i+1] = lambda_n_i_iteration(lamb[i], x[i], dt)
            x[i+1] = x1_i1_iteration_time(lamb[i+1], x[i], dt)
        return x, lamb




