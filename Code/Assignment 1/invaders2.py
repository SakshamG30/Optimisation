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
# from pulp import *
import pulp
import time
import json



def invaders_easy(prob, cooldowns, disabling):
    """Calculate Easy heuristic solution with cooldowns and disabling.

    "cooldowns" and "disabling" are boolean flags to enable additional problem
    considerations.
    """
    # Initialize targeting structure to None
    targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]
    last_fired_time = [-float('inf')] * len(prob.lasers)  # Track the last fired timestep for each laser
    laser_disabled = [False] * len(prob.lasers)  # Track if a laser is disabled
    healths = [ufo.health for ufo in prob.ufos]  # Initialize healths for each UFO

    for t in range(prob.timesteps):
        # Check if any UFO disables a laser
        if disabling:
            for l, laser in enumerate(prob.lasers):
                if laser_disabled[l]:
                    continue  # Skip already disabled lasers
                for u, ufo in enumerate(prob.ufos):
                    ufo_pos = ufo.position(t)
                    if healths[u] > 0 and framework.distance(laser.position(), ufo_pos) <= laser.size:
                        laser_disabled[l] = True  # Disable the laser
                        break  # No need to check other UFOs for this laser

        for l, laser in enumerate(prob.lasers):
            if laser_disabled[l]:
                continue  # Skip targeting if the laser is disabled

            if cooldowns and (t - last_fired_time[l] <= laser.cooldown):
                continue  # Skip this laser if it's still in cooldown

            closest_ufo = None
            min_distance = float('inf')

            # Iterate over all UFOs to find the closest valid target
            for u, ufo in enumerate(prob.ufos):
                ufo_pos = ufo.position(t)
                dist = framework.distance(laser.position(), ufo_pos)
                # print("laser_posssss",laser.position())
                # print("lufoooooooo_posssss",ufo_pos)

                # Check if UFO is a valid target
                if healths[u] > 0 and ufo_pos[1] > 0 and laser.range[0] <= dist <= laser.range[1]:
                    if dist < min_distance:
                        min_distance = dist
                        closest_ufo = u
                #print("health",u,ufo.health)

            

            # Target the closest valid UFO
            if closest_ufo is not None:
                targeting[l][t] = closest_ufo
                
                # Apply damage to the targeted UFO
                healths[closest_ufo] -= laser.damage(prob.ufos[closest_ufo].position(t))
                 # Update the last fired time for this laser
                dead=framework.when_destroyed(prob,targeting,movement=None)
                last_fired_time[l] = t

     # Calculate when each UFO is destroyed
    destruction_times = framework.when_destroyed(prob, targeting)

      # Update the remaining UFO count
    remaining = sum(1 for ufo in healths if ufo > 0)
    return {
        "remaining": remaining,  # Total number of undestroyed UFOs at the end.
        "targeting": targeting,   # Targeting info for lasers.
    }




# def invaders_hard(prob, cooldowns, disabling):
#     """Calculate Hard MILP solution with cooldowns and disabling."""

#     # Initialize the model
#     m = pulp.LpProblem("HardMode", pulp.LpMinimize)
    
#     # Decision Variables: x[i][j][t] is 1 if laser i targets UFO j at time t, 0 otherwise
#     x = [
#         [
#             [
#                 pulp.LpVariable(f"x_{i}{j}{t}", cat="Binary")
#                 for t in range(prob.timesteps)
#             ]
#             for j in range(len(prob.ufos))
#         ]
#         for i in range(len(prob.lasers))
#     ]

#     # Additional decision variables to track if lasers are disabled
#     if disabling:
#         d = [
#             [
#                 pulp.LpVariable(f"d_{i}_{t}", cat="Binary")
#                 for t in range(prob.timesteps)
#             ]
#             for i in range(len(prob.lasers))
#         ]

#     # Objective: Minimize the sum of remaining health for all UFOs
#     total_damage = [
#         pulp.lpSum(
#             x[i][j][t] * prob.lasers[i].damage(prob.ufos[j].position(t))
#             for i in range(len(prob.lasers))
#             for t in range(prob.timesteps)
#         )
#         for j in range(len(prob.ufos))
#     ]

#     remaining_health = pulp.lpSum(
#         [prob.ufos[j].health - total_damage[j] for j in range(len(prob.ufos))]
#     )
#     m += remaining_health, "Minimize_Remaining_Health"

#     # Constraint: Each laser can target at most one UFO per time step
#     for i in range(len(prob.lasers)):
#         for t in range(prob.timesteps):
#             if disabling:
#                 m += pulp.lpSum([x[i][j][t] for j in range(len(prob.ufos))]) <= 1 * (1 - d[i][t]), f"MaxOneTarget_Laser{i}_Time{t}"
#             else:
#                 m += pulp.lpSum([x[i][j][t] for j in range(len(prob.ufos))]) <= 1, f"MaxOneTarget_Laser{i}_Time{t}"

#     # Constraint: UFOs can only be targeted if they are within range
#     for i, laser in enumerate(prob.lasers):
#         for j, ufo in enumerate(prob.ufos):
#             for t in range(prob.timesteps):
#                 dist = framework.distance(laser.pos, ufo.position(t))
#                 if not (laser.range[0] <= dist <= laser.range[1]):
#                     m += x[i][j][t] == 0, f"OutOfRange_Laser{i}_UFO{j}_Time{t}"

#     # Cooldown Constraints (if cooldowns are enabled)
#     if cooldowns:
#         for i, laser in enumerate(prob.lasers):
#             for t in range(prob.timesteps):
#                 for delta in range(1, laser.cooldown + 1):
#                     if t + delta < prob.timesteps:
#                         m += pulp.lpSum(
#                             [x[i][j][t] + x[i][j][t + delta] for j in range(len(prob.ufos))]
#                         ) <= 1, f"Cooldown_Laser{i}_Time{t}_Delta{delta}"

#     # Revised Disabling Constraints (if disabling is enabled)
#     if disabling:
#         for i, laser in enumerate(prob.lasers):
#             for t in range(prob.timesteps):
#                 for j, ufo in enumerate(prob.ufos):
#                     if framework.distance(laser.pos, ufo.position(t)) <= laser.size and ufo.health > 0:
#                         # Laser is disabled if any UFO is within disabling range and active
#                         m += d[i][t] >= x[i][j][t], f"Disablement_Constraint_Laser{i}_Time{t}_UFO{j}"

#                 # Once a laser is disabled, it stays disabled for all future time steps
#                 if t > 0:
#                     m += d[i][t] >= d[i][t-1], f"Persistence_Disablement_Laser{i}_Time{t}"

#     # Solve the MILP with verbose output enabled
#     m.solve(solver=pulp.PULP_CBC_CMD(msg=True))

#     # Check if the solver finished successfully
#     if pulp.LpStatus[m.status] != 'Optimal':
#         raise ValueError("Solver did not find an optimal solution")

#     # Extract the solution
#     targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]
#     for i in range(len(prob.lasers)):
#         for j in range(len(prob.ufos)):
#             for t in range(prob.timesteps):
#                 if pulp.value(x[i][j][t]) > 0.5:
#                     targeting[i][t] = j

#     # Determine destruction times using the when_destroyed function
#     destruction_times = framework.when_destroyed(prob, targeting)

#     # Calculate the remaining UFOs based on whether they were destroyed
#     remaining = sum(1 for d in destruction_times if d is None)
    
#     return {"status": pulp.LpStatus[m.status], "remaining": remaining, "targeting": targeting}


# best new one
def invaders_hard(prob, cooldowns, disabling):
    """Calculate Hard MILP solution.

    "cooldowns" and "disabling" are boolean flags to enable additional problem
    considerations.
    """
    # Initialize the MILP problem
    m = pulp.LpProblem("Invaders_Hard_Mode", pulp.LpMinimize)

    # Decision variables: x[i][t][j] = 1 if laser i targets UFO j at time t, 0 otherwise
    x = {}
    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            for j in range(len(prob.ufos)):
                x[(i, t, j)] = pulp.LpVariable(f"x_{i}_{t}_{j}", cat="Binary")

     # Additional decision variables to track if lasers are disabled
    if disabling:
        d = [
            pulp.LpVariable(f"d_{i}", cat="Binary")
            for i in range(len(prob.lasers))
        ]


    # Introduce binary variables to indicate if a UFO is destroyed
    destroyed = [pulp.LpVariable(f"destroyed_{j}", cat="Binary") for j in range(len(prob.ufos))]

    # Objective: Minimize the number of remaining UFOs at the end of the game
    remaining_ufo = [1 - destroyed[j] for j in range(len(prob.ufos))]
    
    # Add constraints for each UFO
    total_damage_by_ufo = [[] for _ in range(len(prob.ufos))]
    for j in range(len(prob.ufos)):
        ufo = prob.ufos[j]
        total_damage = []
        for t in range(prob.timesteps):
            damage_at_t = []
            for i in range(len(prob.lasers)):
                laser = prob.lasers[i]
                dist = framework.distance(laser.pos, ufo.position(t))
                damage = laser.base_damage + laser.falloff * dist
                damage_at_t.append(damage * x[(i, t, j)])
            total_damage.append(pulp.lpSum(damage_at_t))
            total_damage_by_ufo[j].append(total_damage[-1])
        
        # Sum of damages across all timesteps must be greater than or equal to UFO's health for it to be destroyed
        m += pulp.lpSum(total_damage) >= ufo.health * destroyed[j], f"HealthConstraint_UFO_{j}"
    
    # Ensure that lasers stop targeting a UFO once it is destroyed
    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            for j in range(len(prob.ufos)):
                m += x[(i, t, j)] <= 1 - destroyed[j], f"Stop_Targeting_Destroyed_UFO_{j}_Laser_{i}_Time_{t}"

    # Constraint: UFOs can only be targeted if they are within range
    for i, laser in enumerate(prob.lasers):
        for j, ufo in enumerate(prob.ufos):
            for t in range(prob.timesteps):
                dist = framework.distance(laser.pos, ufo.position(t))
                if not (laser.range[0] <= dist <= laser.range[1]):
                    m += x[(i,t,j)] == 0, f"OutOfRange_Laser{i}_UFO{j}_Time{t}"

    # Cooldown Constraints (if cooldowns are enabled)
    if cooldowns:
        for i, laser in enumerate(prob.lasers):
            for t in range(prob.timesteps - laser.cooldown):
                for j in range(len(prob.ufos)):
                    for k in range(len(prob.ufos)):
                        if j != k:
                            # If a laser switches from targeting UFO j to UFO k, enforce the cooldown
                            m += x[(i, t, j)] + pulp.lpSum(
                                x[(i, t + delta, k)] for delta in range(1, laser.cooldown + 1)
                            ) <= 1, f"Cooldown_Laser{i}_Time{t}_SwitchFrom_{j}_To_{k}"


    # Disabling Constraints (if disabling is enabled)
    if disabling:
        for i, laser in enumerate(prob.lasers):
            for t in range(prob.timesteps):
                for j, ufo in enumerate(prob.ufos):
                    if framework.distance(laser.pos, ufo.position(t)) <= laser.size:
                        # If UFO is within disabling range, disable the laser
                        m += d[i] >= x[(i, t, j)], f"Disablement_Constraint_Laser{i}_Time{t}_UFO{j}"

                # Once a laser is disabled, it stays disabled for all future time steps
                if t > 0:
                    m += d[i] >= d[i], f"Persistence_Disablement_Laser{i}_Time{t}"

    # Ensure that disabled lasers do not target any UFOs
    if disabling:
        for i in range(len(prob.lasers)):
            for t in range(prob.timesteps):
                for j in range(len(prob.ufos)):
                    m += x[(i, t, j)] <= 1 - d[i], f"DisabledLaserCannotTarget_Laser{i}_Time{t}_UFO{j}"
 

    # Ensure that each laser can only target one UFO per timestep
    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            m += pulp.lpSum([x[(i, t, j)] for j in range(len(prob.ufos))]) <= 1, f"SingleTarget_Laser_{i}_Time_{t}"


    # Objective: Minimize the sum of remaining UFOs
    m += pulp.lpSum(remaining_ufo), "MinimizeRemainingUFOs"

    # Solve the MILP
    m.solve(solver=pulp.COIN_CMD(msg=1))
    
    remaining = 0
    # After solving, print health for each UFO at each timestep
    for j in range(len(prob.ufos)):
        ufo = prob.ufos[j]
        print("initial health",ufo.health)
        for t in range(prob.timesteps):
            damage_up_to_t = pulp.value(pulp.lpSum(total_damage_by_ufo[j][:t+1]))
            if damage_up_to_t is None:
                damage_up_to_t = 0  # Handle None case
            current_health = ufo.health - damage_up_to_t
            if current_health <= 0:
                print(f"UFO {j} was destroyed by timestep {t}")
                break
            
            print(f"UFO {j} health at time {t}: {current_health}")
        if current_health > 0:
         remaining += 1

    # Calculate the number of remaining UFOs
    #remaining = sum(pulp.value(remaining_ufo[j]) for j in range(len(prob.ufos)))

    # Extract targeting solution
    targeting = [[None for _ in range(prob.timesteps)] for _ in range(len(prob.lasers))]
    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            for j in range(len(prob.ufos)):
                if pulp.value(x[(i, t, j)]) > 0.5:
                    targeting[i][t] = j

    # Return the results
    return {
        "status": m.status,  # pulp solve model status
        "remaining": remaining,  # Total number of undestroyed UFOs at the end
        "targeting": targeting,  # Targeting info for lasers
    }




def print_invaders_report(instance_name, easy_mode_remaining, hard_mode_remaining, solve_time, root_relaxation_obj, first_feasible_solution, heuristic_gap):
    """Print the results in a formatted table."""
    print(f"Results for Instance: {instance_name}")
    print("-" * 80)
    print(f"Easy Mode Remaining UFOs: {easy_mode_remaining}")
    print(f"Hard Mode Remaining UFOs: {hard_mode_remaining}")
    print(f"MILP Solve Time (s): {solve_time:.2f}")
    print(f"MILP Root Node Linear Relaxation Objective Value: {root_relaxation_obj:.2f}")
    print(f"First Feasible Solution Objective Value: {first_feasible_solution}")
    print(f"Heuristic Optimality Gap (Relative to MILP Root Relaxation): {heuristic_gap:.2%}")
    print("-" * 80)
    print("\n")



def distance(a, b):
    """Manhattan distance between two 2D points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_range(laser, ufo_pos):
    """Check if the UFO is within the laser's range."""
    dist = distance(laser.pos, ufo_pos)
    return laser.range[0] <= dist <= laser.range[1]


def invaders_ultra(prob, max_distance_bound=300):
    """
    Calculate Ultra-Hard MILP solution where lasers can move.

    max_distance_bound: the assumed maximum distance a laser can be from any UFO at any point in time.
    """
    # Initialize the MILP model
    m = pulp.LpProblem("Space_Invader_Ultra_Hard_Mode", pulp.LpMinimize)

    # Variables: x[i][j][t] is 1 if laser i targets UFO j at time t, 0 otherwise
    x = [[[pulp.LpVariable(f"x_{i}_{j}_{t}", cat="Binary") for t in range(prob.timesteps)] 
          for j in range(len(prob.ufos))] for i in range(len(prob.lasers))]

    # Movement variables: position[i][t] represents the (x, y) position of laser i at time t
    
    # position = [[(pulp.LpVariable(f"x_pos_{i}_{t}", lowBound=-max_distance_bound, upBound=max_distance_bound),
    #               pulp.LpVariable(f"y_pos_{i}_{t}", lowBound=-max_distance_bound, upBound=max_distance_bound)) 
    #              for t in range(prob.timesteps)] for i in range(len(prob.lasers))]
    position=[[[0,0] for t in range(prob.timesteps)] for i in range(len(prob.lasers))]
    # print("tttttttttt")
    # print(prob.timesteps)
    for t in range(prob.timesteps):
       for i, laser in enumerate(prob.lasers):
        #    print(" i ", i, " t ", t, "laser ", laser ,"time ", t)
           
           position[i][t][0] = laser.pos[0]+(laser.speed*t)
           position[i][t][1] = laser.pos[1]+(laser.speed*t)
    

    # Auxiliary variables for Manhattan distance
    manhattan_distance = [[[pulp.LpVariable(f"manhattan_dist_{i}_{j}_{t}", lowBound=0) 
                            for t in range(prob.timesteps)] for j in range(len(prob.ufos))] for i in range(len(prob.lasers))]
    abs_diff_x = [[[pulp.LpVariable(f"abs_diff_x_{i}_{j}_{t}", lowBound=0) 
                    for t in range(prob.timesteps)] for j in range(len(prob.ufos))] for i in range(len(prob.lasers))]
    abs_diff_y = [[[pulp.LpVariable(f"abs_diff_y_{i}_{j}_{t}", lowBound=0) 
                    for t in range(prob.timesteps)] for j in range(len(prob.ufos))] for i in range(len(prob.lasers))]

    # Health variables for each UFO at each timestep
    h = { (j, t): pulp.LpVariable(f"h_{j}_{t}", lowBound=0, upBound=prob.ufos[j].health) 
         for j in range(len(prob.ufos)) for t in range(prob.timesteps) }

    # Objective: Minimize the sum of remaining health for all UFOs at the last time step
    m += pulp.lpSum([h[j, prob.timesteps - 1] for j in range(len(prob.ufos))]), "Minimize_UFO_Health"

    # Initial health constraints
    for j, ufo in enumerate(prob.ufos):
        m += h[j, 0] == ufo.health, f"Initial_Health_{j}"

    # Movement constraints for lasers
    for i, laser in enumerate(prob.lasers):
        # Lasers start at their initial positions
        m += position[i][0][0] == laser.pos[0], f"Initial_X_Position_{i}"
        m += position[i][0][1] == laser.pos[1], f"Initial_Y_Position_{i}"

        # Lasers can only move up to their speed limit per timestep
        for t in range(1, prob.timesteps):
            m += (pulp.lpSum([
                abs_diff_x[i][j][t] + abs_diff_y[i][j][t] for j in range(len(prob.ufos))
            ]) <= laser.speed, f"Movement_Constraint_{i}_Time_{t}")

    # Constraints for calculating Manhattan distances
    for i in range(len(prob.lasers)):
        for j in range(len(prob.ufos)):
            for t in range(prob.timesteps):
                # Absolute value constraints for Manhattan distance
                m += abs_diff_x[i][j][t] >= position[i][t][0] - prob.ufos[j].position(t)[0], f"Abs_Diff_X_Positive_{i}_{j}_{t}"
                m += abs_diff_x[i][j][t] >= -(position[i][t][0] - prob.ufos[j].position(t)[0]), f"Abs_Diff_X_Negative_{i}_{j}_{t}"
                m += abs_diff_y[i][j][t] >= position[i][t][1] - prob.ufos[j].position(t)[1], f"Abs_Diff_Y_Positive_{i}_{j}_{t}"
                m += abs_diff_y[i][j][t] >= -(position[i][t][1] - prob.ufos[j].position(t)[1]), f"Abs_Diff_Y_Negative_{i}_{j}_{t}"

                # Manhattan distance is the sum of the absolute differences
                m += manhattan_distance[i][j][t] == abs_diff_x[i][j][t] + abs_diff_y[i][j][t], f"Manhattan_Dist_{i}_{j}_{t}"

                # Lasers can only target UFOs if within max range
                m += manhattan_distance[i][j][t] <= prob.lasers[i].range[1] + (1 - x[i][j][t]) * max_distance_bound, f"MaxRange_Targeting_{i}_{j}_{t}"

    # Health update constraints
    for j, ufo in enumerate(prob.ufos):
        for t in range(1, prob.timesteps):
            laser_pos=[(position[i][t][0]), (position[i][t][1])]
            if laser_pos== [None, None]: laser_pos=[0,0]
            # Calculate the total damage from all lasers targeting this UFO at time t
            damage_sum = pulp.lpSum([x[i][j][t] * prob.lasers[i].damage(prob.ufos[j].position(t), laser_pos) 
                                     for i in range(len(prob.lasers))])
            m += h[j, t] == h[j, t - 1] - damage_sum, f"Health_Update_{j}_{t}"

    # Each laser can target at most one UFO per time step
    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            m += pulp.lpSum([x[i][j][t] for j in range(len(prob.ufos))]) <= 1, f"Single_Target_Per_Laser_{i}_Time_{t}"

    # Solve the MILP model
    m.solve(solver=pulp.COIN_CMD(msg=1))
    

    # Check if the solver finished successfully
    # if pulp.LpStatus[m.status] != 'Optimal':
    #     raise ValueError(f"Solver did not find an optimal solution: Status {pulp.LpStatus[m.status]}")

    # Extract solution
    targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]
    movement = [[laser.pos for _ in range(prob.timesteps)] for laser in prob.lasers]

    for i in range(len(prob.lasers)):
        for j in range(len(prob.ufos)):
            for t in range(prob.timesteps):
                if pulp.value(x[i][j][t]) > 0.5:
                    targeting[i][t] = j

    for i in range(len(prob.lasers)):
        for t in range(prob.timesteps):
            movement[i][t] = (pulp.value(position[i][t][0]), pulp.value(position[i][t][1]))

    remaining = sum(1 for health in [pulp.value(h[j, prob.timesteps - 1]) for j in range(len(prob.ufos))] if health > 0)

    return {
        "status": pulp.LpStatus[m.status],
        "remaining": remaining,
        "targeting": targeting,
        "movement": movement
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
