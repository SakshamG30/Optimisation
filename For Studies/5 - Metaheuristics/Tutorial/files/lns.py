from framework import Problem
import corner_heuristic
import argparse
import random as rand
from copy import deepcopy


def destroy(prob: Problem, fraction_destroyed: float):
    """Destroy a fraction of the containers"""
    number_to_destroy = int(len(prob.conts) * fraction_destroyed)
    # Unimplemented
    pass


def repair(prob: Problem):
    """Repair solution using corner heuristic"""
    # Unimplemented
    pass


def lns(prob_init: Problem,
        n_iters: int,
        fraction_destroyed: float):
    """The LNS algorithm"""
    best = deepcopy(prob_init)
    for i in range(n_iters):
        # Unimplemented
        pass
    return best


if __name__ == "__main__":
    par = argparse.ArgumentParser("2D Packing Large Neighbourhood Search")
    par.add_argument("file", help="json instance file")
    par.add_argument("--save-plot", default=None,
                     help="save plot to file")
    par.add_argument("--no-plot", action='store_true',
                     help="don't plot solution")

    args = par.parse_args()

    rand.seed(0)

    prob = Problem(args.file)
    # You can change the ordering functions and other parameters here:
    prob = lns(prob, n_iters=300, fraction_destroyed=0.5)
    print(prob.objective())
    if not args.no_plot:
        prob.plot(file_name=args.save_plot)
