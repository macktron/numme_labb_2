import numpy as np
import matplotlib.pyplot as plt
import math

N = 1000


dt = 1/N

def alpaha_n(lambda_n):
    return lambda_n**-(3/5)

def U(alpha):
    return -(3/2)*alpha**-(2/3)


def labda_ite_n(lambda_n, x, dt):
    return 