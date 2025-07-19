import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class HvacProblem:
  n: int  # number of rooms along edge, for total of n x n
  steps: int  # number of time steps
  step_dur: float  # duration of a time step (h)
  heat_cap: float  # heat capacity (kWh / degC)
  cond_internal: float  # conductance between internal rooms (kW / degC)
  cond_external: float  # conductance between internal rooms (kW / degC)
  temp_lims: Tuple[float, float]  # lower and upper temperature limits
  pow_max: float  # max power of HVAC system per room (kW)
  price: List[float]  # electricity price for each time step ($ / kW)
  temp_initial: List[float]  # initial temperature for each room (degC)
  temp_external: List[float]  # external temperature for each time step (degC)
  # For each room (ordered left to right, top to bottom), lists the time steps
  # when occupied, where time steps are indexed from 0 to steps - 1.
  # These occupied times require the temperatures to be in their limits.
  occupation: List[List[int]]


@dataclass
class HvacSolution:
    status: str
    objective: float
    temp: List[List[float]]  # for each room, temperature for each time step
    pow: List[List[float]]  # for each room, HVAC power for each time step


def plot_solution(prb: HvacProblem, sol: HvacSolution):
    # Plotting up temperatures, powers and occupancy over time for each room
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    times = [t for t in range(prb.steps)]
    rooms = list(range(prb.n ** 2))
    for r in rooms:
        ax1.plot(times, sol.temp[r], label=r)
        ax2.plot(times, sol.pow[r])
        ax3.plot(prb.occupation[r],
                 [r for _ in range(len(prb.occupation[r]))],
                 ls='none', marker='+')
    ax4.plot(times, prb.price)
    ax1.plot([times[0], times[-1]], [prb.temp_lims[0]]*2, ls='--', color='gray')
    ax1.plot([times[0], times[-1]], [prb.temp_lims[1]]*2, ls='--', color='gray')
    ax2.plot([times[0], times[-1]], [prb.pow_max]*2, ls='--', color='gray')
    ax1.set_ylabel('Room Temp')
    ax2.set_ylabel('Heater Power')
    ax3.set_ylabel('Room Occupied')
    ax4.set_ylabel('Price')
    fig.legend()
    plt.show()
    #fig.savefig('hvac.png')
