import numpy
import matplotlib.pyplot as plt
import math
import time

x = numpy.array([150, 200, 300, 500, 1000, 2000])
y = numpy.array([2, 3, 4, 5, 6, 7])


#a)

#5
def u(a, x):
    return 8*x/(8*a+x)

def u_derivative_a(a, x):
    return -8**2*x/(8*a+x)**2


def least_squard_derivative_a(x, y, a):
    sum = 0
    for i in range(0, len(x)):
        sum += 2*(u(a, x[i]) - y[i])*u_derivative_a(a, x[i])
    return sum

def gradient_descent(x, y, a, learning_rate, iterations):
    for i in range(0, iterations):
        a = a - learning_rate*least_squard_derivative_a(x, y, a)
    return a

def plot_5(x, y, a):
    plt.plot(x, y, 'o')
    x2 = numpy.arange(0, 5000, 0.1)
    plt.plot(x2, u(a,x2), 'g')
    plt.show()

#a = gradient_descent(x, y, 0, 0.1, 100000)
#plot_5(x, y, a)
#print(a)



#6
# linjärt ersättningsproblem
def method1(x, y):
    xlog = numpy.log(x)
    one = numpy.array([1 for x in range(0, len(x))])
    g = numpy.log(8-y)
    A = numpy.array([one, xlog]).T

    
    left = numpy.dot(A.T, A)
    right = numpy.matmul(A.T,g.T)
 
    return numpy.linalg.solve(left, right)



def plot_linear(x, y, a, b):
    x2log = numpy.log(numpy.arange(100, 2050, 1))
    plt.plot(numpy.log(x), numpy.log(8-y), 'o')
    plt.plot(x2log, a + b*x2log, 'g')
    plt.show()


def plot_exp(x, y, a, b):
    plt.plot(x, 8-y, 'o')
    x2 = numpy.arange(1, 5000, 0.1)
    plt.plot(x2, numpy.exp(a+ b*numpy.log(x2)), 'g')
    plt.ylim((0,8))
    plt.show()


#a, b = method1(x, y)
#print(numpy.exp(a), b)
#plot_linear(x, y, a, b)
#plot_exp(x, y, a, b)





#Gauss-Newton
def r(x,y, a, b):
    return 8 - y - a*x**b

def r_derivative_a(x, y, a, b):
    return -x**b

def r_derivative_b(x, y, a, b):
    return -a*x**b*numpy.log(x)





#newton method for a and b of r
def jacobian_matrix_func(x, y, a, b):
    eps = 1e-6
    grad_a = (r(x, y, a+eps, b) - r(x, y, a, b))/eps
    grad_b = (r(x, y, a, b+eps) - r(x, y, a, b))/eps
    return numpy.column_stack([grad_a, grad_b]) 


def gauss_newton(x, y, a, b, iterations):
    old = new = numpy.array([a, b])
    for i in range(0, iterations):
        old = new
        jacobian_matrix = numpy.column_stack([r_derivative_a(x, y, old[0], old[1]), r_derivative_b(x, y, old[0], old[1])])
        new = old - numpy.linalg.inv(jacobian_matrix.T @ jacobian_matrix)@(jacobian_matrix.T @ r(x, y, old[0], old[1]))
        if numpy.linalg.norm(new - old) < 1e-20:
            break
    return new 

def plot_gauss_newton(x, y, a, b):
    plt.plot(x, y, 'o')
    x2 = numpy.arange(1, 5000, 0.1)
    plt.plot(x2,8- a*x2**b, 'g')
    plt.ylim((0,8))
    plt.show()


def error_5(x, y, a):
    sum = 0
    for i in range(len(x)):
        sum += (u(a,x[i])**2 - y[i])**2
    return (sum/len(x))**0.5

def error_6(x, y, a, b):
    sum = 0
    for i in range(len(x)):
        sum += (8-a*x[i]**b-y[i])**2
    return (sum/len(x))**0.5

a,b = gauss_newton(x, y, 5, 0.1, 10000)
print(error_6(x, y, a, b))
#plot_gauss_newton(x, y, a, b)
#print(a, b)

def polynomial_interpolaring(x, y):
    A = numpy.zeros((len(x), len(x)))
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            A[i][j] = x[i]**j
    b = numpy.zeros(len(x))
    for i in range(0, len(x)):
        b[i] = y[i]
    return numpy.linalg.solve(A, b)

def plot_polynomial_interpolaring(x, y, a):
    plt.plot(x, y, 'o')
    x2 = numpy.arange(1, 5000, 0.1)
    y2 = numpy.zeros(len(x2))
    for i in range(0, len(x2)):
        for j in range(0, len(a)):
            y2[i] += a[j]*x2[i]**j
    plt.plot(x2, y2, 'g')
    plt.ylim((0,8))
    plt.show()

#plot_polynomial_interpolaring(x, y,polynomial_interpolaring(x, y))












