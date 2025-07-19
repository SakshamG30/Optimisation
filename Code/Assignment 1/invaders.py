# Copyright subsists in all the material on this repository and vests in the ANU
# (or other parties). Any copying of the material from this site other than for
# the purpose of working on this assignment is a breach of copyright.
#
# In practice, this means that you are not allowed to make any of the existing
# material in this repository nor your assignment available to others (except to
# the course lecturers and tutors) at any time before, during, or after the
# course. In particular, you must keep any clone of this repository and any
# extension of it private at all times.

import argparse
import json
import os
import pulp
from pulp import LpVariable as Var
import framework
from framework import distance


def invaders_easy(prob, cooldowns, disabling):
    """Calculate Easy heuristic solution.

    "cooldowns" and "disabling" are boolean flags to enable additional problem
    considerations.
    """

    remaining_ufos = len(prob.ufos)
    laser_targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]
    ufo_health = [ufo.health for ufo in prob.ufos]  # Track health for each UFO

    # Cooldown and disabling tracking
    cooldown_timers = [0] * len(prob.lasers) if cooldowns else None
    last_targeted_ufo = [None] * len(prob.lasers)  # Track last UFO targeted by each laser
    laser_disabled = [False] * len(prob.lasers) if disabling else None

    for t in range(prob.timesteps):
        for laser_idx, laser in enumerate(prob.lasers):
            if disabling and laser_disabled[laser_idx]:
                # Skip processing if the laser is disabled
                continue

            # Check if any UFO disables the laser
            if disabling:
                for ufo_idx, ufo in enumerate(prob.ufos):
                    ufo_position = ufo.position(t)
                    if ufo_health[ufo_idx] > 0 and distance(laser.position(), ufo_position) <= laser.size:
                        laser_disabled[laser_idx] = True
                        break

            if disabling and laser_disabled[laser_idx]:
                # Skip further processing for this laser if it's disabled
                continue

            if cooldowns and cooldown_timers[laser_idx] > 0:
                # Skip if the laser is on cooldown
                cooldown_timers[laser_idx] -= 1
                continue

            closest_ufo_idx = None
            min_distance = float('inf')

            for ufo_idx, ufo in enumerate(prob.ufos):
                if ufo_health[ufo_idx] > 0 and ufo.position(t)[1] > 0:  # Ensure the UFO is a valid target
                    distance_to_ufo = distance(laser.position(), ufo.position(t))
                    if laser.range[0] <= distance_to_ufo <= laser.range[1]:
                        if distance_to_ufo < min_distance:
                            min_distance = distance_to_ufo
                            closest_ufo_idx = ufo_idx

            if closest_ufo_idx is not None:
                # Target the closest valid UFO
                damage_dealt = laser.damage(prob.ufos[closest_ufo_idx].position(t), laser.position())
                ufo_health[closest_ufo_idx] -= damage_dealt
                laser_targeting[laser_idx][t] = closest_ufo_idx

                if ufo_health[closest_ufo_idx] <= 0:
                    ufo_health[closest_ufo_idx] = 0  # Ensure health does not go negative
                    remaining_ufos -= 1

                if cooldowns:
                    if closest_ufo_idx != last_targeted_ufo[laser_idx]:
                        # Set the cooldown if targeting a different UFO
                        cooldown_timers[laser_idx] = laser.cooldown
                    # Update the last target for this laser
                    last_targeted_ufo[laser_idx] = closest_ufo_idx

    return {
        "remaining": remaining_ufos,
        "targeting": laser_targeting,
    }


def invaders_hard(prob, cooldowns, disabling):
    """Calculate Hard MILP solution."""

    """Calculate Hard MILP solution with cooldowns and disabling options."""

    m = pulp.LpProblem("Hard_Mode", pulp.LpMinimize)

    # Decision variables
    laser_target = {}  # Binary variables for targeting (3D)
    ufo_health = {}  # Continuous variables for ufo health (2D)
    ufo_destroyed = {}  # Binary variables for ufo destroyed (2D)

    # Total base damage of the lasers
    total_base_damage = sum(laser.base_damage for laser in prob.lasers)

    print(f"Total Base Damage: {total_base_damage}")

    # Initialize decision variables
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            for j, ufo in enumerate(prob.ufos):
                laser_target[i, j, t] = pulp.LpVariable(f"laser_target_{i}_{j}_{t}", cat='Binary')

    for j, ufo in enumerate(prob.ufos):
        for t in range(prob.timesteps):
            ufo_health[j, t] = pulp.LpVariable(f"ufo_health_{j}_{t}", lowBound=-total_base_damage, upBound=ufo.health,
                                               cat="Continuous")
            ufo_destroyed[j, t] = pulp.LpVariable(f"ufo_destroyed_{j}_{t}", cat='Binary')

    # Objective: Minimize the sum of final healths (remaining UFOs)
    m += pulp.lpSum([ufo_health[j, prob.timesteps - 1] for j in range(len(prob.ufos))]), "Minimize_UFO_Health"

    # Constraints

    # Initial health constraints for each UFO
    for j, ufo in enumerate(prob.ufos):
        m += ufo_health[j, 0] == ufo.health, f"Initial_Health_{j}"

        for t in range(1, prob.timesteps):
            # Health update: ufo_health[j, t] is ufo_health[j, t-1] minus damage from all targeting lasers
            m += ufo_health[j, t] == ufo_health[j, t - 1] - pulp.lpSum(
                [laser_target[i, j, t] * prob.lasers[i].damage(ufo.position(t)) for i in range(len(prob.lasers))]
            ), f"Health_Update_{j}_{t}"

            # Destruction status: if UFO is destroyed, its health is <= 0
            m += ufo_health[j, t] <= ufo.health * (1 - ufo_destroyed[j, t]), f"Destruction_Status_{j}_{t}"

            # Destruction propagation: once destroyed, remains destroyed
            if t > 0:
                m += ufo_destroyed[j, t] >= ufo_destroyed[j, t - 1], f"Destruction_Propagation_{j}_{t}"

    # Constraint: Each laser can only target one UFO per timestep and can't target destroyed UFOs
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            # Each laser can only target one UFO per timestep
            m += pulp.lpSum([laser_target[i, j, t] for j in range(len(prob.ufos))]) <= 1, f"Single_Target_{i}_{t}"

            # Disable targeting if the laser is disabled
            if disabling:
                for j, ufo in enumerate(prob.ufos):
                    laserdistance = distance(laser.position(), ufo.position(t))
                    if laserdistance <= laser.size:  # Laser is disabled if any UFO is within disabling range
                        # From the time this UFO reaches the laser, it remains disabled for all future time steps
                        for future_t in range(t, prob.timesteps):
                            m += pulp.lpSum([laser_target[i, k, future_t] for k in
                                             range(len(prob.ufos))]) == 0, f"Disable_Laser_{i}_{j}_{t}_{future_t}"

            for j, ufo in enumerate(prob.ufos):
                laserdistance = distance(laser.position(), ufo.position(t))
                # Valid target constraints (must be within range, active, and would receive positive damage)
                if not (laser.range[0] <= laserdistance <= laser.range[1] and prob.lasers[i].damage(
                        ufo.position(t)) > 0):
                    m += laser_target[i, j, t] == 0, f"Out_of_Range_Or_No_Damage_{i}_{j}_{t}"

                # Prevent targeting destroyed UFOs
                m += laser_target[i, j, t] <= 1 - ufo_destroyed[j, t], f"Prevent_Target_Destroyed_{i}_{j}_{t}"

    # Cooldown constraints
    if cooldowns:
        for i, laser in enumerate(prob.lasers):
            for t in range(1, prob.timesteps):
                for j in range(len(prob.ufos)):
                    # Ensure that if a laser targets a new UFO, it must continue targeting it for the duration of the cooldown
                    for t_prime in range(t + 1, min(t + laser.cooldown + 1, prob.timesteps)):
                        m += laser_target[i, j, t_prime] >= laser_target[i, j, t - 1] - (
                                    1 - laser_target[i, j, t]), f"Cooldown_{i}_{j}_{t}_{t_prime}"

    # Solve the problem using the PULP_CBC_CMD solver with verbose output enabled
    m.solve(pulp.PULP_CBC_CMD(msg=1))

    # For Debugging: Checking the health of each UFO at every timestep
    print("UFO Health Status at Each Timestep:")
    for j in range(len(prob.ufos)):
        print(f"UFO {j}:")
        for t in range(prob.timesteps):
            health_value = ufo_health[j, t].value()
            print(f"  Time {t}: Health = {health_value:.2f}")

    # Extract the solution
    remaining = sum(1 for j in range(len(prob.ufos)) if ufo_health[j, prob.timesteps - 1].value() > 0)
    targeting = [[None for _ in range(prob.timesteps)] for _ in range(len(prob.lasers))]

    for t in range(prob.timesteps):
        for i in range(len(prob.lasers)):
            for j in range(len(prob.ufos)):
                if laser_target[i, j, t].value() == 1:
                    targeting[i][t] = j

    return {
        "status": m.status,  # Solver model status
        "remaining": remaining,  # Number of UFOs remaining with health > 0
        "targeting": targeting,  # Targeting info for lasers
    }


def invaders_ultra(prob):
    """Calculate Ultra-Hard MILP solution."""

    m = pulp.LpProblem("Ultra_Hard_Mode", pulp.LpMinimize)
    #
    # # Decision variables
    # x = {}
    # p = {}
    # h = {}
    # d = {}
    #
    # total_base_damage = sum(laser.base_damage for laser in prob.lasers)
    #
    # # Initialize decision variables
    # for t in range(prob.timesteps):
    #     for i, laser in enumerate(prob.lasers):
    #         p[i, t] = (pulp.LpVariable(f"p_{i}_{t}_x", lowBound=-300, upBound=300, cat="Continuous"),
    #                    pulp.LpVariable(f"p_{i}_{t}_y", lowBound=-300, upBound=300, cat="Continuous"))
    #         for j, ufo in enumerate(prob.ufos):
    #             x[i, j, t] = pulp.LpVariable(f"x_{i}_{j}_{t}", cat='Binary')
    #
    # for j, ufo in enumerate(prob.ufos):
    #     for t in range(prob.timesteps):
    #         h[j, t] = pulp.LpVariable(f"h_{j}_{t}", lowBound=-total_base_damage, upBound=ufo.health, cat="Continuous")
    #         d[j, t] = pulp.LpVariable(f"d_{j}_{t}", cat='Binary')
    #
    # m += pulp.lpSum([h[j, prob.timesteps - 1] for j in range(len(prob.ufos))]), "Minimize_UFO_Health"
    #
    # # Constraints
    #
    # # Initial health constraints for each UFO
    # for j, ufo in enumerate(prob.ufos):
    #     m += h[j, 0] == ufo.health, f"Initial_Health_{j}"
    #
    #     for t in range(1, prob.timesteps):
    #         # Health update: h[j, t] is h[j, t - 1] minus damage from all targeting lasers
    #         m += h[j, t] == h[j, t - 1] - pulp.lpSum(
    #             [x[i, j, t] * prob.lasers[i].damage(ufo.position(t), laser_pos=(p[i, t][0], p[i, t][1]), ignore_min_range=True)
    #              for i in range(len(prob.lasers))]
    #         ), f"Health_Update_{j}_{t}"
    #
    #         # Destruction status: if UFO is destroyed, its health is <= 0
    #         m += h[j, t] <= ufo.health * (1 - d[j, t]), f"Destruction_Status_{j}_{t}"
    #
    #         # Destruction propagation: once destroyed, remains destroyed
    #         if t > 0:
    #             m += d[j, t] >= d[j, t - 1], f"Destruction_Propagation_{j}_{t}"
    #
    # # Constraint: Each laser can only target one UFO per timestep and can't target destroyed UFOs
    # for t in range(prob.timesteps):
    #     for i, laser in enumerate(prob.lasers):
    #         # Each laser can only target one UFO per timestep
    #         m += pulp.lpSum([x[i, j, t] for j in range(len(prob.ufos))]) <= 1, f"Single_Target_{i}_{t}"
    #
    #         for j, ufo in enumerate(prob.ufos):
    #             # Valid target constraints (must be within range, active, and would receive positive damage)
    #             m += x[i, j, t] <= 1 - d[j, t], f"Prevent_Target_Destroyed_{i}_{j}_{t}"
    #
    # # Laser movement constraints: Ensuring that lasers move within their speed limits
    # for i, laser in enumerate(prob.lasers):
    #     for t in range(1, prob.timesteps):
    #         manhattan_distance = (abs(p[i, t][0] - p[i, t - 1][0]) + abs(p[i, t][1] - p[i, t - 1][1]))
    #         m += manhattan_distance <= laser.speed, f"Speed_Limit_{i}_{t}"
    #
    # # Solve the problem using the PULP_CBC_CMD solver with verbose output enabled
    m.solve(pulp.PULP_CBC_CMD(msg=1))
    #
    # # Extract the solution
    # remaining = sum(1 for j in range(len(prob.ufos)) if h[j, prob.timesteps - 1].value() > 0)
    # targeting = [[None for _ in range(prob.timesteps)] for _ in range(len(prob.lasers))]
    # movement = [[None for _ in range(prob.timesteps)] for _ in range(len(prob.lasers))]
    #
    remaining = len(prob.ufos)
    targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]
    movement = [[laser.pos
                 for _ in range(prob.timesteps)]
                for laser in prob.lasers]

    # for t in range(prob.timesteps):
    #     for i in range(len(prob.lasers)):
    #         for j in range(len(prob.ufos)):
    #             if x[i, j, t].value() == 1:
    #                 targeting[i][t] = j
    #         movement[i][t] = (p[i, t][0].value(), p[i, t][1].value())
    #
    return {
        "status": m.status,  # Solver model status
        "remaining": remaining,  # Number of UFOs remaining with health > 0
        "targeting": targeting,  # Targeting info for lasers
        "movement": movement,  # Movement info for lasers
    }




if __name__ == "__main__":
    par = argparse.ArgumentParser("Space Invasion")
    par.add_argument("file", help="json instance file")
    par.add_argument("--cooldowns", action="store_true",
                     help="enable handling of cooldowns")
    par.add_argument("--disabling", action="store_true",
                     help="enable handling of disabling")
    par.add_argument("--animate", action="store_true",
                     help="create animation of solution")

    args = par.parse_args()

    prob = framework.Problem(args.file)

    # Running solvers
    easy_sol = invaders_easy(prob,
                             cooldowns=args.cooldowns,
                             disabling=args.disabling)
    hard_sol = invaders_hard(prob,
                             cooldowns=args.cooldowns,
                             disabling=args.disabling)
    ultra_sol = invaders_ultra(prob)

    # Printing results
    print(f"Easy Remaining {easy_sol['remaining']} / {len(prob.ufos)}")
    print(f"Hard Status {hard_sol['status']}")
    print(f"Hard Remaining {hard_sol['remaining']} / {len(prob.ufos)}")
    print(f"Ultra Status {ultra_sol['status']}")
    print(f"Ultra Remaining {ultra_sol['remaining']} / {len(prob.ufos)}")

    # Solutions to file
    fn_base = os.path.splitext(args.file)[0]
    json.dump(easy_sol, open(fn_base + "-easy.json", "w"), indent=2)
    json.dump(hard_sol, open(fn_base + "-hard.json", "w"), indent=2)
    json.dump(ultra_sol, open(fn_base + "-ultra.json", "w"), indent=2)

    # Visualise solution
    framework.plot(prob, easy_sol["targeting"], fn_base + "-easy.png")
    framework.plot(prob, hard_sol["targeting"], fn_base + "-hard.png")
    framework.plot(prob,
                   ultra_sol["targeting"],
                   fn_base + "-ultra.png",
                   movement=ultra_sol["movement"])
    if args.animate:
        framework.animate(prob, easy_sol["targeting"], fn_base + "-easy.mp4")
        framework.animate(prob, hard_sol["targeting"], fn_base + "-hard.mp4")
        framework.animate(prob,
                          ultra_sol["targeting"],
                          fn_base + "-ultra.mp4",
                          movement=ultra_sol["movement"])
