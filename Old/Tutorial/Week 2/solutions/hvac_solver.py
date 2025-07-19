import pulp
from pulp import LpVariable as Var
import numpy as np
from typing import Any, Union

import sys
import os
# Adding the folder with hvac.py and hvac_instances.py released with the questions to the folders
# searched for importing
questions_file_folder = os.path.abspath(os.path.join('..', 'files'))
sys.path.append(questions_file_folder)

import hvac
import hvac_instances


def get_cond(prb: hvac.HvacProblem, i: int, j: int) -> float:
    """Internal / external wall conductance depending on valid room index"""
    n = prb.n
    if i >= 0 and i < n and j >= 0 and j < n:
        return prb.cond_internal
    return prb.cond_external


def get_temp(prb: hvac.HvacProblem, temp: Any, i: int, j: int) -> Any:
    """Room / external temperatures for each time step depending on if valid room index"""
    n = prb.n
    if i >= 0 and i < n and j >= 0 and j < n:
        return temp[i, j]
    return np.array(prb.temp_external)


def solve_hvac(prb: hvac.HvacProblem) -> hvac.HvacSolution:
    m = pulp.LpProblem()
    n = prb.n
    n_rooms = n ** 2
    steps = prb.steps
    step_dur = prb.step_dur

    # Room variables are shaped into 3D array with n x n grid and time for
    # third dimension. This makes indexing easier.
    # Temperatures for each room over time
    temp = np.array([Var(f'temp_{r}_{t}')
                     for r in range(n_rooms) for t in range(steps)]
                    ).reshape(n, n, steps)
    # HVAC powers for each room over time
    pow_max = prb.pow_max
    # 1 d)
    #pow_max = 2
    pow = np.array([Var(f'p_hvac_{r}_{t}', 0.0, pow_max)
                       for r in range(n * n) for t in range(steps)]
                   ).reshape(n, n, steps)

    # 2 a)
    #temp_error = np.array([Var(f'temp_error_{r}_{t}')
    #                       for r in range(n * n) for t in range(steps)]
    #                      ).reshape(n, n, steps)
    #soft_price = 0.12  # $ / degC h
    #temp_preferred = 20

    for i in range(n):
        for j in range(n):
            # Summing together the heat into room from hvac and walls
            p = pow[i, j].copy()
            p += get_cond(prb, i - 1, j) * (
                get_temp(prb, temp, i - 1, j) - temp[i, j])  # West
            p += get_cond(prb, i + 1, j) * (
                get_temp(prb, temp, i + 1, j) - temp[i, j])  # East
            p += get_cond(prb, i, j - 1) * (
                get_temp(prb, temp, i, j - 1) - temp[i, j])  # North
            p += get_cond(prb, i, j + 1) * (
                get_temp(prb, temp, i, j + 1) - temp[i, j])  # South

            # Apply the temperature state update over time
            temp_prev = prb.temp_initial[i * n + j]
            for t in range(steps):
                m += p[t] * step_dur == prb.heat_cap * (temp[i, j, t] - temp_prev)
                temp_prev = temp[i, j, t]
                # 2 a)
                #m += temp_error[i, j, t] >= temp[i, j, t] - temp_preferred
                #m += temp_error[i, j, t] >= temp_preferred - temp[i, j, t]

    # Setting temperature limits when occupied
    # Reshaping as don't need grid structure any more
    temp = temp.reshape(n_rooms, steps)
    pow = pow.reshape(n_rooms, steps)
    soft = 0  # soft constraint penalty (for part 2)
    # 2 a)
    #temp_error = temp_error.reshape(n_rooms, steps)
    for r in range(n_rooms):
        for t in prb.occupation[r]:
            m += temp[r, t] >= prb.temp_lims[0]
            m += temp[r, t] <= prb.temp_lims[1]
            # 2 a)
            #soft += temp_error[r, t] * soft_price * step_dur
        # 1 b)
        #t_start = int(1 / step_dur)  # steps to first hour
        #for t in range(t_start, steps):
        #    m += temp[r, t] >= prb.temp_lims[0]
        #    m += temp[r, t] <= prb.temp_lims[1]

    # Cost is price times sum of power used in each room
    m += pulp.lpSum(prb.price[t] * step_dur * pulp.lpSum(pow[:, t])
             for t in range(steps)) + soft


    m.solve()
    # 2 a)
    #print(f'Elec: {m.objective.value() - soft.value()} Soft: {soft.value()}')

    extract_value = np.vectorize(lambda x: x.value())
    return hvac.HvacSolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        temp=extract_value(temp),
        pow=extract_value(pow),
    )


if __name__ == "__main__":
    prb = hvac_instances.INSTANCE1X1
    #prb = hvac_instances.INSTANCE6X6

    sol = solve_hvac(prb)

    print(f'Status {sol.status}, Cost {sol.objective}')
    hvac.plot_solution(prb, sol)
