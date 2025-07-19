#!/usr/bin/env python3

import sys
import pulp
from pulp import LpVariable as Var
import numpy as np


def pigeonhole_sum_over_pigeons(pigeons, pigeonholes):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Equal to 1 if pigeon i is in hole j
    x = np.array([
        Var('x_{}_{}'.format(i, j), cat='Binary') for i in range(pigeons)
        for j in range(pigeonholes)]).reshape((pigeons, pigeonholes))

    # Each pigeon is in exactly one pigeonhole
    for i in range(pigeons):
        m += x[i, :].sum() == 1

    # Each pigeonhole has at most one pigeon
    for j in range(pigeonholes):
        m += x[:, j].sum() <= 1

    m += 0

    m.solve(solver=pulp.COIN_CMD(msg=1))
    extract_value = np.vectorize(lambda x: x.value())
    return {'status': m.status,
            'x': extract_value(x),
            }


def pigeonhole_pairwise_sum(pigeons, pigeonholes):
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Equal to 1 if pigeon i is in hole j
    x = np.array([
        Var('x_{}_{}'.format(i, j), cat='Binary') for i in range(pigeons)
        for j in range(pigeonholes)]).reshape((pigeons, pigeonholes))

    # Each pigeon is in exactly one pigeonhole
    for i in range(pigeons):
        m += x[i, :].sum() == 1

    # Each pigeonhole has at most one pigeon
    for j in range(pigeonholes):
        for i in range(pigeons):
            for k in range(pigeons):
                if i != k:
                    m += x[i, j] + x[k, j] <= 1

    m += 0

    m.solve(solver=pulp.COIN_CMD(msg=1))
    extract_value = np.vectorize(lambda x: x.value())
    return {'status': m.status,
            'x': extract_value(x),
            }

if __name__ == "__main__":
    constr_dispatcher = {"sum-over-pigeons": pigeonhole_sum_over_pigeons,
                    "pairwise-sum": pigeonhole_pairwise_sum
                    }
    if len(sys.argv) != 4 or (len(sys.argv) > 1 and sys.argv[1] not in constr_dispatcher):
        print("Usage: pigeonhole [sum-over-pigeons|pairwise-sum] num-pigeons num-pigeonhole")
        sys.exit(-1)
    pigeons = int(sys.argv[2])
    pigeonholes = int(sys.argv[3])
    res = constr_dispatcher[sys.argv[1]](pigeons, pigeonholes)
    print(res)
