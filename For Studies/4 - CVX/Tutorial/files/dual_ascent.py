import argparse
import json
import numpy as np


def gradient_descent(prob, x_init=None, max_iter=1000, default_step=0.9,
                     tol=1e-9, verbose=False):
    # If not explicitly provided, initialise to zeros.
    x = np.zeros(prob.n) if x_init is None else x_init.copy()

    # TO IMPLEMENT

    return {
        "x": x,  # variables
        "objective": prob.objective(x),  # objective value
        "iter": None,  # number of iterations taken
        "within_tolerance": False,  # was tolerance satisfied
    }


def dual_ascent(prob, x_init=None, mu_init=None, max_iter=1000,
                default_step=0.9, tol=1e-9, verbose=False):
    # If not explicitly provided, initialise to zeros.
    x = np.zeros(prob.n) if x_init is None else x_init.copy()
    mu = np.zeros(prob.m) if mu_init is None else mu_init.copy()

    # TO IMPLEMENT

    return {
        "x": x,  # primal variables
        "mu": mu,  # dual variables
        "objective": prob.objective(x),  # objective value
        "iter": None,  # number of iterations in dual ascent
        "sub_iter": None,  # total number of iterations take in sub-problems
        "within_tolerance": False,  # was tolerance satisfied
    }


if __name__ == "__main__":
    par = argparse.ArgumentParser("Dual Gradient Ascent")
    par.add_argument("file", nargs="?", help="json instance file")
    par.add_argument("-v", action="store_true", help="verbose solver output")

    args = par.parse_args()
    import problems

    if args.file is None:  # solve the hard-coded instances

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
    else:  # solve the provided QCQP instance
        print(f"QCQP {args.file}")
        prob = problems.GenericQCQP(json.load(open(args.file)))
        res = dual_ascent(prob, verbose=args.v)
        print(f"Dual ascent: {res}")
