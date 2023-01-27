
def f(x ,a, b):
    return a*x + b*x**2

def least_squard(x, y, a, b):
    sum = 0
    for i in range(0, len(x)):
        sum += (f(x[i], a, b) - y[i])**2
    return sum

def least_squard_derivative_a(x, y, a, b):
    sum = 0
    for i in range(0, len(x)):
        sum += 2*(f(x[i], a, b) - y[i])*x[i]
    return sum

def least_squard_derivative_b(x, y, a, b):
    sum = 0
    for i in range(0, len(x)):
        sum += 2*(f(x[i], a, b) - y[i])*x[i]**2
    return sum

def gradient_descent(x, y, a, b, learning_rate, iterations):
    for i in range(0, iterations):
        a = a - learning_rate*least_squard_derivative_a(x, y, a, b)
        b = b - learning_rate*least_squard_derivative_b(x, y, a, b)
    return a, b

#a, b = gradient_descent(x, y, 0, 0, 0.0001, 100000)
#print(a, b)




 