import pulp
from pulp import LpVariable as Var


def dual():
    # Create model with default solver
    m = pulp.LpProblem(sense=pulp.LpMaximize)

    # Create unbounded variables
    v1 = Var('v1', lowBound=0)
    v2 = Var('v2', lowBound=0)
    v3 = Var('v3', lowBound=0)

    # Add constraints to model
    m += v1 - v2 == 1
    m += 2 * v1 + v2 == 2
    m += 3 * v1 - 2 * v2 - v3 == -5

    # Set objective of model
    m += 2 * v1 - 6 * v3

    m.solve(solver=pulp.COIN_CMD(msg=1))

    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Obj: {m.objective.value()}')
    print(f'v1, v2, v3: {(v1.value(), v2.value(), v3.value())}')


if __name__ == "__main__":
    dual()
