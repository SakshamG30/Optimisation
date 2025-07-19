import argparse
import json
import numpy as np


class Lagrangian:
    def __init__(self, prob, mu):
        """
        Takes a lagrangian relaxation of a constrained problem with provided
        dual variables.
        """
        self.prob = prob
        self.mu = mu

    def objective(self, x):
        return self.prob.objective(x) + self.mu.dot(self.prob.constraints(x))

    def objective_grad(self, x):
        return (self.prob.objective_grad(x)
                + self.mu.dot(self.prob.constraints_jac(x)))


def bb_step_1(x, xp, g, gp, tol):
    """First rule-based step size method from Barzilai and Borwein 1988.

    Returns None if rule can't be applied.
    """
    if gp is None:  # no previous gradient
        return None
    g_diff = g - gp
    if np.linalg.norm(g_diff) <= tol:
        return None  # nearly linear
    return (x - xp).dot(g_diff) / g_diff.dot(g_diff)


def gradient_descent(prob, x_init=None, max_iter=1000, default_step=0.9,
                     tol=1e-9, verbose=False):
    x = np.zeros(prob.n) if x_init is None else x_init.copy()
    g = None
    xp = None  # previous iterate
    gp = None  # previous gradient
    within_tolerance = False
    for i in range(max_iter):
        gp = g
        g = prob.objective_grad(x)  # calculate new gradient for x
        if np.linalg.norm(g) <= tol:
            within_tolerance = True
            break  # gradient is nearly zero, job done
        step = bb_step_1(x, xp, g, gp, tol)
        step = default_step if step is None else step  #  fall back on default
        xp = x
        x = xp - step * g  # update iterate
        if verbose:
            print(f"GD {i}: {(x, xp, g, gp)}")

    return {
        "x": x,
        "objective": prob.objective(x),
        "iter": i + 1,
        "within_tolerance": within_tolerance,
    }


def dual_ascent(prob, x_init=None, mu_init=None, max_iter=1000,
                default_step=0.9, tol=1e-9, verbose=False):
    x = np.zeros(prob.n) if x_init is None else x_init.copy()
    mu = np.zeros(prob.m) if mu_init is None else mu_init.copy()
    g = None  # dual function gradient
    xp = None  # previous primal iterate
    mup = None  # previous dual iterate
    gp = None  # previous dual function gradient
    clamp_at_zero = np.vectorize(lambda e: max(e, 0.0))
    within_tolerance = False
    sub_iter = 0  # inner loop iterations
    for i in range(max_iter):
        # Using latest mu
        lag_prob = Lagrangian(prob, mu)  # form problem for lagrangian function
        xp = x
        # Being lazy not checking if subproblem solved to within tolerance
        gd_res = gradient_descent(lag_prob, xp, max_iter, default_step, tol,
                                  verbose)
        x = gd_res["x"]
        sub_iter += gd_res["iter"]
        errors = kkt(prob, x, mu)  # get error in kkt conditions for mu, x pair
        if sum(errors) <= tol:
            within_tolerance = True
            break  # errors in kkt nearly zero, job done
        gp = g
        g = prob.constraints(x)  # dual function gradient is just constraints
        step = bb_step_1(mu, mup, g, gp, tol)
        # Taking abs because ascent not descent...
        step = default_step if step is None else abs(step)
        mup = mu
        mu = clamp_at_zero(mup + step * g)  # mu's must remain positive
        if verbose:
            print(f"DA {i}: {(x, xp, mu, mup, g, gp, errors)}")

    return {
        "x": x,
        "mu": mu,
        "objective": prob.objective(x),
        "iter": i + 1,
        "sub_iter": sub_iter,
        "within_tolerance": within_tolerance,
    }


def kkt(prob, x, mu):
    """Calculate errors in satisifying kkt conditions"""
    when_positive = np.vectorize(lambda e: max(e, 0.0))
    when_negative = np.vectorize(lambda e: min(e, 0.0))
    stationarity = prob.objective_grad(x) + mu.dot(prob.constraints_jac(x))
    primary = when_positive(prob.constraints(x))
    dual = when_negative(mu)
    complementary = mu.dot(prob.constraints(x))

    return (np.linalg.norm(stationarity),
            np.linalg.norm(primary),
            np.linalg.norm(dual),
            np.linalg.norm(complementary),
            )


if __name__ == "__main__":
    par = argparse.ArgumentParser("Dual Gradient Ascent")
    par.add_argument("file", nargs="?", help="json instance file")
    par.add_argument("-v", action="store_true", help="verbose solver output")

    args = par.parse_args()
    import problems

    if args.file is None:
        # Problem 1
        print("Problem 1")
        prob = problems.Problem1()
        res = gradient_descent(prob, verbose=args.v)
        print(f"Gradient descent: {res}")
        res = dual_ascent(prob, verbose=args.v)
        print(f"Dual ascent: {res}")

        # Problem 2
        print("Problem 2")
        prob = problems.Problem2()
        res = gradient_descent(prob, verbose=args.v)
        print(f"Gradient descent: {res}")
        res = dual_ascent(prob, verbose=args.v)
        print(f"Dual ascent: {res}")
    else:
        print(f"QCQP {args.file}")
        prob = problems.GenericQCQP(json.load(open(args.file)))
        res = dual_ascent(prob, verbose=args.v)
        print(f"Dual ascent: {res}")
