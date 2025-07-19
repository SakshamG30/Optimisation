from framework import Problem
import corner_heuristic
import argparse
import os
import random as rand
from copy import deepcopy


def destroy(prob: Problem, fraction_destroyed: float):
    """Destroy a fraction of the containers"""
    number_to_destroy = int(len(prob.conts) * fraction_destroyed)
    for cont in rand.sample(prob.conts, number_to_destroy):
        prob.unpacked += cont.unpack_all()


def repair(prob: Problem):
    """Repair solution using corner heuristic"""
    corner_heuristic.corner_heuristic(
        prob,
        order_boxes=lambda x: sorted(x, reverse=True, key=lambda y: y.weight),
        order_conts=lambda x: rand.sample(x, len(x)),
        order_corners=lambda x: rand.sample(x, len(x)),
        order_orients=lambda x: rand.sample(x, len(x)),
    )


def lns(prob_init: Problem,
        n_iters: int,
        fraction_destroyed: float):
    """The LNS algorithm"""
    best = deepcopy(prob_init)
    for i in range(n_iters):
        prob = deepcopy(best)
        destroy(prob, fraction_destroyed)
        repair(prob)
        if prob.objective() > best.objective():
            best = prob
            print(f"New best solution {i}: {best.objective()}")
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
    prob = lns(prob, n_iters=300, fraction_destroyed=0.5)
    print(prob.objective())
    if not args.no_plot:
        prob.plot(file_name=args.save_plot)
