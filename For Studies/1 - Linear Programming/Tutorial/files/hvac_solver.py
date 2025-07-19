import pulp
from pulp import LpVariable as Var
import numpy as np

import hvac
import hvac_instances


def solve_hvac(prb: hvac.HvacProblem) -> hvac.HvacSolution:
    m = pulp.LpProblem()
    n_rooms = prb.n ** 2

    # Hint: it might be easier to store variables associated with rooms in a
    # numpy array

    m += 0

    m.solve()
    return hvac.HvacSolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        temp=[[0.0] * prb.steps] * n_rooms,  # TO BE REPLACED
        pow=[[0.0] * prb.steps] * n_rooms,  # TO BE REPLACED
    )


if __name__ == "__main__":
    prb = hvac_instances.INSTANCE1X1
    #prb = hvac_instances.INSTANCE6X6

    sol = solve_hvac(prb)

    print(f'Status {sol.status}, Cost {sol.objective}')
    hvac.plot_solution(prb, sol)
