import numpy
import matplotlib.pyplot as plt


x = numpy.array([0, 0.5, 1, 1.5, 2, 2.99, 3])
y = numpy.array([0, 0.52, 1.09, 1.75, 2.45, 3.5, 4])



def least_squard(x, y):
    A = numpy.array([x, x**2])
    left = numpy.dot(A, A.T)
    right = numpy.dot(A, y)
    b= numpy.linalg.solve(left, right)
    return numpy.array([0, b[0], b[1]])


def polynomial_interpolation(x, y):
    A = numpy.array([x**i for i in range(len(x))]).T #Naiva ansatsen --> dim(p(x)) = n-1
    return numpy.linalg.solve(A, y)

def plot(x, y, a, b):
    plt.plot(x, y, 'o')
    x2 = numpy.arange(0, 3, 0.01)
    plt.plot(x2, a[0] + a[1]*x2 + a[2]*x2**2, 'r')
    plt.plot(x2, b[0] + b[1]*x2 + b[2]*x2**2 + b[3]*x2**3 + b[4]*x2**4 + b[5]*x2**5 + b[6]*x2**6, 'g')
    plt.show()


def error_a(x, y, a):
    sum = 0
    for i in range(len(x)):
        sum += (a[0] + a[1]*x[i] + a[2]*x[i]**2 - y[i])**2
    return (sum/len(x))**0.5

def error_b(x, y, b):
    sum = 0
    for i in range(len(x)):
        sum += (b[0] + b[1]*x[i] + b[2]*x[i]**2 + b[3]*x[i]**3 + b[4]*x[i]**4 + b[5]*x[i]**5 + b[6]*x[i]**6 - y[i])**2
    return (sum/len(x))**0.5


a = least_squard(x, y)
b = polynomial_interpolation(x, y)
print(b,a)
print(error_a(x, y, a), error_b(x, y, b))


plot(x, y, a, b)






