import numpy as np
import math


class Problem1:
    """ 1D problem:
    min (x + 2)^2
    s.t. (x - 1)(x - 6) <= 0
    """

    def __init__(self):
        self.n = 1
        self.m = 1

    def objective(self, x):
        return (x[0] + 2.0) ** 2

    def objective_grad(self, x):
        return np.array([2.0 * (x[0] + 2.0)])

    def constraints(self, x):
        return np.array([(x[0] - 1.0) * (x[0] - 6.0)])

    def constraints_jac(self, x):
        return np.array([2.0 * x[0] - 7.0])


class Problem2:
    """ 2D problem:
    min 2e^(x - 1) + 3e^(-4y - 2) + x^2 + y^2
    s.t. (3x - 2y)^2 - 5 <= 0
    s.t. -x + 5 <= 0
    """

    def __init__(self):
        self.n = 2
        self.m = 2

    def objective(self, x):
        return (2.0 * math.e ** (x[0] - 1.0)
                + 3.0 * math.e ** (-4.0 * x[1] - 2.0)
                + x[0] ** 2
                + x[1] ** 2
                )

    def objective_grad(self, x):
        return np.array([
            2.0 * math.e ** (x[0] - 1.0) + 2.0 * x[0],
            -12.0 * math.e ** (-4.0 * x[1] - 2.0) + 2.0 * x[1],
        ])

    def constraints(self, x):
        return np.array([
            (3.0 * x[0] - 2.0 * x[1]) ** 2 - 5.0,
            -x[0] + 5.0,
        ])

    def constraints_jac(self, x):
        return np.array([
            [6.0 * (3.0 * x[0] - 2.0 * x[1]),
             -4.0 * (3.0 * x[0] - 2.0 * x[1])],
            [-1.0, 0.0],
        ])


def symmetric_from_lower_triangle(a):
    """ Takes a list of lists that represent the lower left triangle entries
    for a symmetric matrix and returns the resulting symmetric numpy array.
    """
    n = len(a)
    # Expand out rows to will upper triangle with zeros
    a = np.array([row + [0.0] * (n - len(row)) for row in a])
    # Fill in upper triangle with lower values
    return a + a.T - np.diag(a.diagonal())


class QuadraticConstraint:
    """ Constraint of the form:
    0.5 x^T P_i x + a_i^T x - b_i <= 0
    """

    def __init__(self, data):
        self.P = symmetric_from_lower_triangle(data["P"])
        self.a = np.array(data["a"])
        self.b = data["b"]


class GenericQCQP:
    """ A Generic quadratically constrained quadratic program.

    min 0.5 x^T Q x + c^T x
    s.t. 0.5 x^T P_i x + a_i^T x - b_i <= 0   for all i in 1, ..., m

    where Q and P_i are all symmetric
    """

    def __init__(self, data):
        self.n = len(data["Q"])  # Number of variables
        self.m = len(data["constraints"])  # Number of constraints
        self.Q = symmetric_from_lower_triangle(data["Q"])
        self.c = np.array(data["c"])
        self.cons = [QuadraticConstraint(con) for con in data["constraints"]]

    def objective(self, x):
        return 0.5 * x.dot(self.Q).dot(x) + self.c.dot(x)

    def objective_grad(self, x):
        return self.Q.dot(x) + self.c

    def constraints(self, x):
        return np.array([
            0.5 * x.dot(con.P).dot(x) + con.a.dot(x) - con.b
            for con in self.cons
        ])

    def constraints_jac(self, x):
        return np.array([
            con.P.dot(x) + con.a
            for con in self.cons
        ])
