#!/usr/bin/env python3

import argparse
import pulp
from pulp import LpVariable as Var
from pulp import lpSum, lpDot


def problem(land_allocation=None, yield_modifier=None):
    crop_yield = [2.5, 3, 20]
    planting_cost = [150, 230, 260]
    selling_price = [170, 150,  36, 10]
    purchase_price = [238, 210]
    min_required = [200, 240]
    beets_quota = 6000

    if yield_modifier is not None:
        crop_yield = [yield_modifier * y for y in crop_yield]

    m = pulp.LpProblem(sense=pulp.LpMinimize)

    allocation = []
    if land_allocation is not None:
        assert len(land_allocation) == len(planting_cost)
        allocation = land_allocation
    else:
        allocation = [Var(f'x_{i+1}', lowBound=0) for i in range(0, len(crop_yield))]
    buy = [Var(f'y_{i+1}', lowBound=0) for i in range(0, len(purchase_price))]
    sell = [Var(f'w_{i+1}', lowBound=0) for i in range(0, len(selling_price))]

    m += lpSum(allocation) <= 500
    for i in range(0, len(purchase_price)):
        m += crop_yield[i] * allocation[i] + buy[i] - sell[i] >= min_required[i]

    m += sell[2] + sell[3] <= crop_yield[2] * allocation[2]
    m += sell[2] <= beets_quota

    m += lpDot(planting_cost, allocation) + lpDot(purchase_price, buy) - lpDot(selling_price, sell)
    print(m)
    m.solve()

    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Obj: {m.objective.value()}')
    if land_allocation is None:
        print('Land allocation:', [x.value() for x in allocation])
    print('Amount bought:', [y.value() for y in buy])
    print('Amount sold:', [w.value() for w in sell])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--allocation',
                        type=lambda s: [float(item) for item in s.split(',')],
                        help='Comma-separated allocation for wheat, corn and beets',
                        default=None)
    parser.add_argument('--yield-mod',
                        type=float,
                        help='Ratio (e.g., 0.8 and 1.2) for the yield of all crops',
                        default=None)
    args = parser.parse_args()
    print(f"Forced Allocation: {args.allocation}")
    print(f"Yield Modfier: {args.yield_mod}")

    assert args.allocation is None or len(args.allocation) == 3, \
            "Either do not provide a land allocation or provide the allocation for all crops"

    problem(land_allocation=args.allocation, yield_modifier=args.yield_mod)
