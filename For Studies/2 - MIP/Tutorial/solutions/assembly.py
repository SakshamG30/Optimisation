import pulp
from pulp import LpVariable as Var
from pulp import lpSum
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set


@dataclass
class Job:
    duration: int  # duration of job
    arms: int  # number of arms required for job
    elevated: bool = False  # whether car needs to be elevated to perform job
    lowered: bool = False  # whether car needs to be lowered to perform job


@dataclass
class AssemblyProblem:
    arms: int  # number of robot arms available
    jobs: Dict[str, Job]
    ordering: List[Tuple[str, str]]  # ordering between pairs of jobs
    mutex: List[Set[str]]  # sets of jobs that can't run at same time


@dataclass
class AssemblySolution:
    status: str  # optimisation status
    objective: float  # objective value
    start_times: Dict[str, int]  # start time of each job


def get_start_time(inds):
    """The start time for solved indicator variables."""
    for t, j in enumerate(inds):
        if 0.99 <= j.value() <= 1.01:
            return t
    print('No valid start time')
    return None


def assembly_a(prb: AssemblyProblem,
             steps: int,
             max_solve_sec: int) -> AssemblySolution:
    """
    Model that only accounts for arm limits.

    Only a start time indicator variable is used.
    """
    m = pulp.LpProblem()

    # indicator variable for the start time of each job
    jobs_inds = {k: [Var(f'j_{k}_{t}', cat='Binary') for t in range(steps)]
                 for k in prb.jobs}
    # the overall makespan of the schedule
    makespan = Var('makespan', cat='Integer')

    # each job runs exactly once
    for inds in jobs_inds.values():
        m += lpSum(inds) == 1

    # enforce number of arms
    for t in range(steps):
        arms = 0
        for k, job in prb.jobs.items():
            # work out the range of start times for this job which would lead
            # to the use of arm(s) at t
            t_fr = max(0, t + 1 - job.duration)
            t_to = t + 1  # one past the end
            # if the job is started during this period, enforce arm usage
            arms += job.arms * lpSum(jobs_inds[k][t_fr: t_to])
        m += arms <= prb.arms

    # the makespan is greater than the completion time of each job
    for k, job in prb.jobs.items():
        for t in range(steps):
            m += makespan >= t * jobs_inds[k][t] + job.duration
    m += makespan

    # Setting a maximum solve time
    m.solve(pulp.COIN(maxSeconds=max_solve_sec))
    return AssemblySolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        start_times={k: get_start_time(j) for k, j in jobs_inds.items()},
    )


def assembly_b(prb: AssemblyProblem,
             steps: int,
             max_solve_sec: int) -> AssemblySolution:
    """
    Model that only accounts for arm limits.

    A job running variable is added.
    """
    m = pulp.LpProblem()

    # indicator variable for the start time of each job
    jobs_inds = {k: [Var(f'j_{k}_{t}', cat='Binary') for t in range(steps)]
                 for k in prb.jobs}
    # indicator variable for when job is running
    jobs_run = {k: [Var(f'j_run_{k}_{t}', cat='Binary') for t in range(steps)]
                for k in prb.jobs}
    # the overall makespan of the schedule
    makespan = Var('makespan', cat='Integer')

    # each job runs exactly once
    for inds in jobs_inds.values():
        m += lpSum(inds) == 1

    # link the job running indicator with start time indicator variable
    for t in range(steps):
        for k, job in prb.jobs.items():
            # work out the range of start times for this job which would lead
            # to it running at t
            t_fr = max(0, t + 1 - job.duration)
            t_to = t + 1
            # if the job is started during this period, running status
            m += jobs_run[k][t] == lpSum(jobs_inds[k][t_fr: t_to])

    # enforce number of arms
    for t in range(steps):
        arms = 0
        for k, job in prb.jobs.items():
            # job is running, enforce use of arms
            arms += job.arms * jobs_run[k][t]
        m += arms <= prb.arms

    # the makespan is greater than the completion time of each job
    for k, job in prb.jobs.items():
        for t in range(steps):
            m += makespan >= t * jobs_inds[k][t] + job.duration
    m += makespan

    # Setting a maximum solve time
    m.solve(pulp.COIN(maxSeconds=max_solve_sec))
    return AssemblySolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        start_times={k: get_start_time(j) for k, j in jobs_inds.items()},
    )


def assembly_c(prb: AssemblyProblem,
             steps: int,
             max_solve_sec: int) -> AssemblySolution:
    """
    Model that only accounts for arm limits.

    A job running variable is added.
    A job start time variable is added.
    """
    m = pulp.LpProblem()

    # indicator variable for the start time of each job
    jobs_inds = {k: [Var(f'j_{k}_{t}', cat='Binary') for t in range(steps)]
                 for k in prb.jobs}
    # indicator variable for when job is running
    jobs_run = {k: [Var(f'j_run_{k}_{t}', cat='Binary') for t in range(steps)]
                for k in prb.jobs}
    # start time of each job
    jobs_start = {k: Var(f'j_start_{k}', cat='Integer') for k in prb.jobs}
    # the overall makespan of the schedule
    makespan = Var('makespan', cat='Integer')

    # each job runs exactly once
    for inds in jobs_inds.values():
        m += lpSum(inds) == 1

    # link the job running indicator with start time indicator variable
    for t in range(steps):
        for k, job in prb.jobs.items():
            # work out the range of start times for this job which would lead
            # to it running at t
            t_fr = max(0, t + 1 - job.duration)
            t_to = t + 1
            # if the job is started during this period, running status
            m += jobs_run[k][t] == lpSum(jobs_inds[k][t_fr: t_to])

    # link the start time indicator variable with start time variable
    for k, inds in jobs_inds.items():
        m += jobs_start[k] == sum(t * x for t, x in enumerate(inds))

    # enforce number of arms
    for t in range(steps):
        arms = 0
        for k, job in prb.jobs.items():
            # job is running, enforce use of arms
            arms += job.arms * jobs_run[k][t]
        m += arms <= prb.arms

    # the makespan is greater than the completion time of each job
    for k, job in prb.jobs.items():
        m += makespan >= jobs_start[k] + job.duration
    m += makespan

    # Setting a maximum solve time
    m.solve(pulp.COIN(maxSeconds=max_solve_sec))
    return AssemblySolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        start_times={k: get_start_time(j) for k, j in jobs_inds.items()},
    )


def assembly_d(prb: AssemblyProblem,
             steps: int,
             max_solve_sec: int) -> AssemblySolution:
    """
    Model that accounts for arm limits and orderings.
    """
    m = pulp.LpProblem()

    # indicator variable for the start time of each job
    jobs_inds = {k: [Var(f'j_{k}_{t}', cat='Binary') for t in range(steps)]
                 for k in prb.jobs}
    # indicator variable for when job is running
    jobs_run = {k: [Var(f'j_run_{k}_{t}', cat='Binary') for t in range(steps)]
                for k in prb.jobs}
    # start time of each job
    jobs_start = {k: Var(f'j_start_{k}', cat='Integer') for k in prb.jobs}
    # the overall makespan of the schedule
    makespan = Var('makespan', cat='Integer')

    # each job runs exactly once
    for inds in jobs_inds.values():
        m += lpSum(inds) == 1

    # link the job running indicator with start time indicator variable
    for t in range(steps):
        for k, job in prb.jobs.items():
            # work out the range of start times for this job which would lead
            # to it running at t
            t_fr = max(0, t + 1 - job.duration)
            t_to = t + 1
            # if the job is started during this period, running status
            m += jobs_run[k][t] == lpSum(jobs_inds[k][t_fr: t_to])

    # link the start time indicator variable with start time variable
    for k, inds in jobs_inds.items():
        m += jobs_start[k] == sum(t * x for t, x in enumerate(inds))

    # enforce number of arms
    for t in range(steps):
        arms = 0
        for k, job in prb.jobs.items():
            # job is running, enforce use of arms
            arms += job.arms * jobs_run[k][t]
        m += arms <= prb.arms

    # enfore job orderings
    for k1, k2 in prb.ordering:
        m += jobs_start[k1] + prb.jobs[k1].duration <= jobs_start[k2]

    # the makespan is greater than the completion time of each job
    for k, job in prb.jobs.items():
        m += makespan >= jobs_start[k] + job.duration
    m += makespan

    # Setting a maximum solve time
    m.solve(pulp.COIN(maxSeconds=max_solve_sec))
    return AssemblySolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        start_times={k: get_start_time(j) for k, j in jobs_inds.items()},
    )


def assembly_e(prb: AssemblyProblem,
             steps: int,
             max_solve_sec: int) -> AssemblySolution:
    """
    Model that accounts for arm limits, orderings, elevation and mutexes.
    """
    m = pulp.LpProblem()

    # indicator variable for the start time of each job
    jobs_inds = {k: [Var(f'j_{k}_{t}', cat='Binary') for t in range(steps)]
                 for k in prb.jobs}
    # indicator variable for when job is running
    jobs_run = {k: [Var(f'j_run_{k}_{t}', cat='Binary') for t in range(steps)]
                for k in prb.jobs}
    # start time of each job
    jobs_start = {k: Var(f'j_start_{k}', cat='Integer') for k in prb.jobs}
    # the overall makespan of the schedule
    makespan = Var('makespan', cat='Integer')
    # elevation whether or not the car is elevated
    elev = [Var(f'e_{t}', cat='Binary') for t in range(steps)]

    # each job runs exactly once
    for inds in jobs_inds.values():
        m += lpSum(inds) == 1

    # link the job running indicator with start time indicator variable
    for t in range(steps):
        for k, job in prb.jobs.items():
            # work out the range of start times for this job which would lead
            # to it running at t
            t_fr = max(0, t + 1 - job.duration)
            t_to = t + 1
            # if the job is started during this period, running status
            m += jobs_run[k][t] == lpSum(jobs_inds[k][t_fr: t_to])

    # link the start time indicator variable with start time variable
    for k, inds in jobs_inds.items():
        m += jobs_start[k] == sum(t * x for t, x in enumerate(inds))

    # enforce number of arms
    for t in range(steps):
        arms = 0
        for k, job in prb.jobs.items():
            # job is running, enforce use of arms
            arms += job.arms * jobs_run[k][t]
        m += arms <= prb.arms

    # enfore job orderings
    for k1, k2 in prb.ordering:
        m += jobs_start[k1] + prb.jobs[k1].duration <= jobs_start[k2]

    # elevation restrictions
    for k, job in prb.jobs.items():
        if job.elevated:
            for t in range(steps):
                m += jobs_run[k][t] <= elev[t]
        if job.lowered:
            for t in range(steps):
                m += jobs_run[k][t] <= 1 - elev[t]

    # mutex restrictions
    for mut_jobs in prb.mutex:
        for t in range(steps):
            # at most one running at a time
            m += lpSum(jobs_run[k][t] for k in mut_jobs) <= 1

    # the makespan is greater than the completion time of each job
    for k, job in prb.jobs.items():
        m += makespan >= jobs_start[k] + job.duration
    m += makespan

    # Setting a maximum solve time
    m.solve(pulp.COIN(maxSeconds=max_solve_sec))
    return AssemblySolution(
        status=pulp.LpStatus[m.status],
        objective=m.objective.value(),
        start_times={k: get_start_time(j) for k, j in jobs_inds.items()},
    )


INSTANCE = AssemblyProblem(
  arms=3,
  jobs={
    "gearbox": Job(duration=2, arms=2, elevated=True),
    "engine": Job(duration=5, arms=3, lowered=True),
    "bonnet": Job(duration=2, arms=2, lowered=True),
    "axle_f": Job(duration=2, arms=2, elevated=True),
    "axle_b": Job(duration=2, arms=2, elevated=True),
    "wheel_f_l": Job(duration=1, arms=1),
    "wheel_f_r": Job(duration=1, arms=1),
    "wheel_b_l": Job(duration=1, arms=1),
    "wheel_b_r": Job(duration=1, arms=1),
    "exhaust": Job(duration=2, arms=2, elevated=True),
    "under_carriage": Job(duration=3, arms=3, elevated=True),
    "dashboard": Job(duration=3, arms=3, lowered=True),
    "windscreen": Job(duration=1, arms=1, lowered=True),
    "door_f_l": Job(duration=2, arms=2),
    "door_f_r": Job(duration=2, arms=2),
    "door_b_l": Job(duration=2, arms=2),
    "door_b_r": Job(duration=2, arms=2),
    "boot": Job(duration=1, arms=2),
    "seat_f_l": Job(duration=2, arms=2),
    "seat_f_r": Job(duration=2, arms=2),
    "seat_b": Job(duration=3, arms=2),
    "lights_f": Job(duration=1, arms=1),
    "lights_b": Job(duration=1, arms=1)
  },
  ordering=[
    ("gearbox", "engine"),
    ("engine", "axle_f"),
    ("axle_f", "wheel_f_l"),
    ("axle_f", "wheel_f_r"),
    ("axle_b", "wheel_b_l"),
    ("axle_b", "wheel_b_r"),
    ("gearbox", "under_carriage"),
    ("exhaust", "under_carriage"),
    ("engine", "bonnet"),
    ("dashboard", "windscreen"),
    ("dashboard", "door_f_l"),
    ("dashboard", "door_f_r"),
    ("seat_f_l", "door_f_l"),
    ("seat_f_r", "door_f_r"),
    ("seat_b", "door_b_l"),
    ("seat_b", "door_b_r")
  ],
  mutex=[
    set(["engine", "bonnet", "dashboard", "windscreen"]),
  ]
)


if __name__ == "__main__":
    prb = INSTANCE
    sol = assembly_a(prb, steps=60, max_solve_sec=60)
    #sol = assembly_b(prb, steps=60, max_solve_sec=60)
    #sol = assembly_c(prb, steps=60, max_solve_sec=60)
    #sol = assembly_d(prb, steps=60, max_solve_sec=60)
    #sol = assembly_e(prb, steps=60, max_solve_sec=60)
    # f)
    #prb.arms = 4
    #sol = assembly_e(prb, steps=60, max_solve_sec=60)
    # g)
    #prb.arms = 4
    #sol = assembly_e(prb, steps=30, max_solve_sec=60)

    print(f'Status {sol.status}, Objective {sol.objective}')

    fig, ax1 = plt.subplots(1, sharex=True)
    bar_data = {'x': [], 'bottom': [], 'height': [], 'tick_label': []}
    for i, n in enumerate(sorted(prb.jobs.keys())):
        t = sol.start_times[n]
        bar_data['x'].append(i)
        bar_data['bottom'].append(t)
        bar_data['height'].append(prb.jobs[n].duration)
        bar_data['tick_label'].append(n)
    ax1.bar(**bar_data)
    ax1.set_xlabel('Jobs')
    ax1.set_ylabel('Time (minutes)')
    ax1.tick_params(axis='x', rotation=90)
    #fig.tight_layout()
    plt.show()
