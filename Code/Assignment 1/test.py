# Claude Cooldown
# m = pulp.LpProblem(sense=pulp.LpMinimize)
#
#     # Decision Variables
#     x = {(l, u, t): pulp.LpVariable(f'x_{l}_{u}_{t}', cat='Binary')
#          for l in range(len(prob.lasers))
#          for u in range(len(prob.ufos))
#          for t in range(prob.timesteps)}
#
#     health = {(u, t): pulp.LpVariable(f'health_{u}_{t}', lowBound=0, cat='Continuous')
#               for u in range(len(prob.ufos))
#               for t in range(prob.timesteps + 1)}
#
#     alive = {u: pulp.LpVariable(f'alive_{u}', cat='Binary')
#              for u in range(len(prob.ufos))}
#
#     # New variables for cooldown
#     if cooldowns:
#         fired = {(l, t): pulp.LpVariable(f'fired_{l}_{t}', cat='Binary')
#                  for l in range(len(prob.lasers))
#                  for t in range(prob.timesteps)}
#
#     # Objective: Minimize the number of remaining UFOs
#     m += pulp.lpSum(alive[u] for u in range(len(prob.ufos)))
#
#     # Constraints
#     for u in range(len(prob.ufos)):
#         m += health[u, 0] == prob.ufos[u].health
#         for t in range(prob.timesteps):
#             m += health[u, t + 1] == health[u, t] - pulp.lpSum(
#                 x[l, u, t] * prob.lasers[l].damage(prob.ufos[u].position(t))
#                 for l in range(len(prob.lasers))
#             )
#         m += health[u, prob.timesteps] <= prob.ufos[u].health * alive[u]
#         m += health[u, prob.timesteps] >= alive[u]
#
#     for l in range(len(prob.lasers)):
#         for t in range(prob.timesteps):
#             m += pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos))) <= 1
#
#             for u in range(len(prob.ufos)):
#                 ufo_pos = prob.ufos[u].position(t)
#                 laser_pos = prob.lasers[l].position()
#                 if (distance(laser_pos, ufo_pos) < prob.lasers[l].range[0] or
#                         distance(laser_pos, ufo_pos) > prob.lasers[l].range[1] or
#                         ufo_pos[1] <= 0):
#                     m += x[l, u, t] == 0
#
#             # Cooldown constraints
#             if cooldowns:
#                 m += fired[l, t] == pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos)))
#                 if t >= prob.lasers[l].cooldown:
#                     m += pulp.lpSum(fired[l, k] for k in range(t - prob.lasers[l].cooldown + 1, t + 1)) <= 1
#
#     # Solve the model
#     solver = pulp.PULP_CBC_CMD(msg=1)
#     m.solve(solver)
#
#     # Extract solution
#     remaining = sum(1 for u in range(len(prob.ufos)) if alive[u].value() > 0.5)
#     targeting = [[next((u for u in range(len(prob.ufos)) if x[l, u, t].value() > 0.5), None)
#                   for t in range(prob.timesteps)]
#                  for l in range(len(prob.lasers))]
#
#     return {
#         "status": pulp.LpStatus[m.status],
#         "remaining": remaining,
#         "targeting": targeting,
#     }


# Claude Disabling
# m = pulp.LpProblem(sense=pulp.LpMinimize)
#
#     # Decision Variables
#     x = {(l, u, t): pulp.LpVariable(f'x_{l}_{u}_{t}', cat='Binary')
#          for l in range(len(prob.lasers))
#          for u in range(len(prob.ufos))
#          for t in range(prob.timesteps)}
#
#     health = {(u, t): pulp.LpVariable(f'health_{u}_{t}', lowBound=0, cat='Continuous')
#               for u in range(len(prob.ufos))
#               for t in range(prob.timesteps + 1)}
#
#     alive = {u: pulp.LpVariable(f'alive_{u}', cat='Binary')
#              for u in range(len(prob.ufos))}
#
#     if cooldowns:
#         fired = {(l, t): pulp.LpVariable(f'fired_{l}_{t}', cat='Binary')
#                  for l in range(len(prob.lasers))
#                  for t in range(prob.timesteps)}
#
#     if disabling:
#         laser_active = {(l, t): pulp.LpVariable(f'laser_active_{l}_{t}', cat='Binary')
#                         for l in range(len(prob.lasers))
#                         for t in range(prob.timesteps)}
#
#     # Objective: Minimize the number of remaining UFOs
#     m += pulp.lpSum(alive[u] for u in range(len(prob.ufos)))
#
#     # Constraints
#     for u in range(len(prob.ufos)):
#         m += health[u, 0] == prob.ufos[u].health
#         for t in range(prob.timesteps):
#             m += health[u, t + 1] == health[u, t] - pulp.lpSum(
#                 x[l, u, t] * prob.lasers[l].damage(prob.ufos[u].position(t))
#                 for l in range(len(prob.lasers))
#             )
#         m += health[u, prob.timesteps] <= prob.ufos[u].health * alive[u]
#         m += health[u, prob.timesteps] >= alive[u]
#
#     for l in range(len(prob.lasers)):
#         if disabling:
#             m += laser_active[l, 0] == 1  # All lasers start active
#
#         for t in range(prob.timesteps):
#             if disabling:
#                 # Laser can only fire if it's active
#                 m += pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos))) <= laser_active[l, t]
#
#                 # Laser becomes inactive if touched by a UFO
#                 for u in range(len(prob.ufos)):
#                     ufo_pos = prob.ufos[u].position(t)
#                     laser_pos = prob.lasers[l].position()
#                     if distance(laser_pos, ufo_pos) <= prob.lasers[l].size:
#                         if t > 0:
#                             m += laser_active[l, t] <= laser_active[l, t - 1] - alive[u]
#                         else:
#                             m += laser_active[l, t] <= 1 - alive[u]
#
#                 # Once inactive, stays inactive
#                 if t > 0:
#                     m += laser_active[l, t] <= laser_active[l, t - 1]
#
#             for u in range(len(prob.ufos)):
#                 ufo_pos = prob.ufos[u].position(t)
#                 laser_pos = prob.lasers[l].position()
#                 if (distance(laser_pos, ufo_pos) < prob.lasers[l].range[0] or
#                         distance(laser_pos, ufo_pos) > prob.lasers[l].range[1] or
#                         ufo_pos[1] <= 0):
#                     m += x[l, u, t] == 0
#
#             # Cooldown constraints
#             if cooldowns:
#                 m += fired[l, t] == pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos)))
#                 if t >= prob.lasers[l].cooldown:
#                     m += pulp.lpSum(fired[l, k] for k in range(t - prob.lasers[l].cooldown + 1, t + 1)) <= 1
#
#     # Solve the model
#     solver = pulp.PULP_CBC_CMD(msg=1)
#     m.solve(solver)
#
#     # Extract solution
#     remaining = sum(1 for u in range(len(prob.ufos)) if alive[u].value() > 0.5)
#     targeting = [[next((u for u in range(len(prob.ufos)) if x[l, u, t].value() > 0.5), None)
#                   for t in range(prob.timesteps)]
#                  for l in range(len(prob.lasers))]
#
#     return {
#         "status": pulp.LpStatus[m.status],
#         "remaining": remaining,
#         "targeting": targeting,
#     }


#Best Code so far:
m = pulp.LpProblem("HardMode", pulp.LpMinimize)

    # Decision variables
    x = {}  # Binary variables: x[i, j, t] = 1 if laser i targets UFO j at time t
    h = {}  # Continuous variables: h[j, t] is the health of UFO j at time t
    d = [pulp.LpVariable(f"d_{i}", cat='Binary') for i in range(len(prob.lasers))]  # Disabling status of lasers

    # Initialize decision variables
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            for j, ufo in enumerate(prob.ufos):
                x[i, j, t] = pulp.LpVariable(f"x_{i}_{j}_{t}", cat='Binary')

    for j, ufo in enumerate(prob.ufos):
        for t in range(prob.timesteps):
            h[j, t] = pulp.LpVariable(f"h_{j}_{t}", lowBound=0, upBound=ufo.health)

    # Objective: Minimize the sum of final healths (remaining UFOs)
    m += pulp.lpSum([h[j, prob.timesteps - 1] for j in range(len(prob.ufos))]), "Minimize_UFO_Health"

    # Initial health constraints for each UFO
    for j, ufo in enumerate(prob.ufos):
        m += h[j, 0] == ufo.health, f"Initial_Health_{j}"

    # Update health constraints and targeting logic
    for t in range(1, prob.timesteps):
        for j, ufo in enumerate(prob.ufos):
            # Health update: health at time t should equal health at t-1 minus the damage from all targeting lasers
            m += h[j, t] == h[j, t - 1] - pulp.lpSum(
                [x[i, j, t] * prob.lasers[i].damage(ufo.position(t)) for i in range(len(prob.lasers))]
            ), f"Health_Update_{j}_{t}"

    # Laser targeting constraints
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            # Each laser can target at most one UFO per timestep
            m += pulp.lpSum([x[i, j, t] for j in range(len(prob.ufos))]) <= 1, f"Single_Target_{i}_{t}"

            for j, ufo in enumerate(prob.ufos):
                dist = framework.distance(laser.position(), ufo.position(t))
                valid_target = (laser.range[0] <= dist <= laser.range[1]) and (ufo.position(t)[1] > 0)
                if not valid_target:
                    m += x[i, j, t] == 0, f"Invalid_Target_{i}_{j}_{t}"

    # Cooldown constraints
    if cooldowns:
        for i, laser in enumerate(prob.lasers):
            for t in range(1, prob.timesteps):
                for j in range(len(prob.ufos)):
                    # Enforce cooldown if the laser switches to a new target
                    for t_prime in range(t + 1, min(t + laser.cooldown + 1, prob.timesteps)):
                        m += x[i, j, t] <= x[i, j, t - 1] + (1 - pulp.lpSum(
                            x[i, k, t_prime] for k in range(len(prob.ufos)))), f"Cooldown_{i}_{j}_{t}_{t_prime}"

    # Disabling constraints
    if disabling:
        for i, laser in enumerate(prob.lasers):
            for t in range(prob.timesteps):
                for j, ufo in enumerate(prob.ufos):
                    ufo_pos = ufo.position(t)
                    # If a UFO disables a laser, set the disable status
                    if distance(laser.position(), ufo_pos) <= laser.size:
                        m += d[i] == 1, f"Disable_Laser_{i}_{t}_{j}"

                # If a laser is disabled, it cannot target any UFOs
                for j in range(len(prob.ufos)):
                    m += x[i, j, t] <= (1 - d[i]), f"Disable_Target_{i}_{j}_{t}"

    # Solve the MILP problem
    m.solve(pulp.PULP_CBC_CMD(msg=1))

    # Process the results
    remaining = sum(1 for j in range(len(prob.ufos)) if h[j, prob.timesteps - 1].value() > 0)
    targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]

    for t in range(prob.timesteps):
        for i in range(len(prob.lasers)):
            for j in range(len(prob.ufos)):
                if x[i, j, t].value() == 1:
                    targeting[i][t] = j

    return {
        "status": m.status,  # Solver model status
        "remaining": remaining,  # Number of UFOs remaining with health > 0
        "targeting": targeting,  # Targeting info for lasers
    }






# m = pulp.LpProblem("UFO_Destruction", pulp.LpMinimize)
#
#     # Decision variables
#     x = pulp.LpVariable.dicts("target",
#                               ((l, u, t) for l in range(len(prob.lasers))
#                                for u in range(len(prob.ufos))
#                                for t in range(prob.timesteps)),
#                               cat='Binary')
#
#     health = pulp.LpVariable.dicts("health",
#                                    ((u, t) for u in range(len(prob.ufos))
#                                     for t in range(prob.timesteps + 1)),
#                                    lowBound=0, cat='Continuous')
#
#     alive = {u: pulp.LpVariable(f'alive_{u}', cat='Binary')
#              for u in range(len(prob.ufos))}
#
#     if cooldowns:
#         fired = {(l, t): pulp.LpVariable(f'fired_{l}_{t}', cat='Binary')
#                  for l in range(len(prob.lasers))
#                  for t in range(prob.timesteps)}
#
#     if disabling:
#         laser_active = {(l, t): pulp.LpVariable(f'laser_active_{l}_{t}', cat='Binary')
#                         for l in range(len(prob.lasers))
#                         for t in range(prob.timesteps)}
#
#     # Objective: Minimize the number of remaining UFOs
#     m += pulp.lpSum(alive[u] for u in range(len(prob.ufos)))
#
#     # Constraints
#     # 1. Each laser can target at most one UFO at a time
#     for l in range(len(prob.lasers)):
#         for t in range(prob.timesteps):
#             m += pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos))) <= 1
#
#     # 2. Cooldown period for lasers
#     if cooldowns:
#         for l in range(len(prob.lasers)):
#             for t in range(1, prob.timesteps):
#                 m += fired[l, t] == pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos)))
#                 if t >= prob.lasers[l].cooldown:
#                     m += pulp.lpSum(fired[l, k] for k in range(t - prob.lasers[l].cooldown + 1, t + 1)) <= 1
#
#     # 3. Disable lasers if they are within disabling range of a UFO
#     if disabling:
#         for l in range(len(prob.lasers)):
#             for t in range(prob.timesteps):
#                 for u in range(len(prob.ufos)):
#                     ufo_pos = prob.ufos[u].position(t)
#                     laser_pos = prob.lasers[l].position()
#                     if distance(laser_pos, ufo_pos) <= prob.lasers[l].size:
#                         m += laser_active[l, t] <= 1 - alive[u]
#                 # A laser can only fire if it is active
#                 m += pulp.lpSum(x[l, u, t] for u in range(len(prob.ufos))) <= laser_active[l, t]
#
#     # 4. Health and alive status of UFOs
#     for u in range(len(prob.ufos)):
#         m += health[u, 0] == prob.ufos[u].health
#         for t in range(prob.timesteps):
#             m += health[u, t + 1] == health[u, t] - pulp.lpSum(
#                 x[l, u, t] * prob.lasers[l].damage(prob.ufos[u].position(t))
#                 for l in range(len(prob.lasers))
#             )
#         m += health[u, prob.timesteps] <= prob.ufos[u].health * alive[u]
#         m += health[u, prob.timesteps] >= alive[u]
#
#     # Solve the model
#     solver = pulp.PULP_CBC_CMD(msg=1)
#     m.solve(solver)
#
#     # Extract solution
#     remaining = sum(1 for u in range(len(prob.ufos)) if alive[u].value() > 0.5)
#     targeting = [[next((u for u in range(len(prob.ufos)) if x[l, u, t].value() > 0.5), None)
#                   for t in range(prob.timesteps)]
#                  for l in range(len(prob.lasers))]
#
#     return {
#         "status": pulp.LpStatus[m.status],
#         "remaining": remaining,
#         "targeting": targeting,
#     }


#Arslan's code
m = pulp.LpProblem("Space_Invader_Hard_Mode", pulp.LpMinimize)

    # Decision variables
    x = {}  # Binary variables: x[i, j, t] = 1 if laser i targets UFO j at time t
    h = {}  # Continuous variables: h[j, t] is the health of UFO j at time t
    d = {}  # Binary variables: d[j, t] = 1 if UFO j is destroyed at or before time t

    # Initialize decision variables
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            for j, ufo in enumerate(prob.ufos):
                x[i, j, t] = pulp.LpVariable(f"x_{i}_{j}_{t}", cat='Binary')

    for j, ufo in enumerate(prob.ufos):
        for t in range(prob.timesteps):
            h[j, t] = pulp.LpVariable(f"h_{j}_{t}", lowBound=0, upBound=ufo.health, cat="Continuous")
            d[j, t] = pulp.LpVariable(f"d_{j}_{t}", cat='Binary')

    # Objective: Minimize the sum of final healths (remaining UFOs)
    m += pulp.lpSum([h[j, prob.timesteps - 1] for j in range(len(prob.ufos))]), "Minimize_UFO_Health"

    # Constraints

    # Initial health constraints for each UFO
    for j, ufo in enumerate(prob.ufos):
        m += h[j, 0] == ufo.health, f"Initial_Health_{j}"

        for t in range(1, prob.timesteps):
            # Health update: h[j, t] is h[j, t-1] minus damage from all targeting lasers
            m += h[j, t] == h[j, t - 1] - pulp.lpSum(
                [x[i, j, t] * prob.lasers[i].damage(ufo.position(t)) for i in range(len(prob.lasers))]
            ), f"Health_Update_{j}_{t}"

            # Destruction status: if UFO is destroyed, its health is <= 0
            m += h[j, t] <= ufo.health * (1 - d[j, t]), f"Destruction_Status_{j}_{t}"

            # Destruction propagation: once destroyed, remains destroyed
            if t > 0:
                m += d[j, t] >= d[j, t - 1], f"Destruction_Propagation_{j}_{t}"

    # Constraint: Each laser can only target one UFO per timestep and can't target destroyed UFOs
    for t in range(prob.timesteps):
        for i, laser in enumerate(prob.lasers):
            m += pulp.lpSum([x[i, j, t] for j in range(len(prob.ufos))]) <= 1, f"Single_Target_{i}_{t}"

            for j, ufo in enumerate(prob.ufos):
                distance = framework.distance(laser.position(), ufo.position(t))
                # Valid target constraints (must be within range, active, and would receive positive damage)
                if not (laser.range[0] <= distance <= laser.range[1] and prob.lasers[i].damage(ufo.position(t)) > 0):
                    m += x[i, j, t] == 0, f"Out_of_Range_Or_No_Damage_{i}_{j}_{t}"

                # Prevent targeting destroyed UFOs
                m += x[i, j, t] <= 1 - d[j, t], f"Prevent_Target_Destroyed_{i}_{j}_{t}"

    # Cooldown constraints (optional)
    if cooldowns:
        cooldown_remaining = [[0 for _ in range(prob.timesteps)] for _ in prob.lasers]
        for i, laser in enumerate(prob.lasers):
            for t in range(prob.timesteps):
                if t > 0:
                    for j, ufo in enumerate(prob.ufos):
                        if t - laser.cooldown > 0:
                            m += x[i, j, t] + pulp.lpSum(x[i, k, t - laser.cooldown - 1] for k in range(len(prob.ufos))) <= 1 + x[i, j, t-1], f"Cooldown_Constraint_{i}_{j}_{t}"

    # Solve the problem using the PULP_CBC_CMD solver with verbose output enabled
    m.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=30))

    # Check if the solver found an optimal solution
    if pulp.LpStatus[m.status] != 'Optimal':
        raise ValueError("Solver did not find an optimal solution")

    # Extract the solution
    remaining = sum(1 for j in range(len(prob.ufos)) if h[j, prob.timesteps - 1].value() > 0)
    targeting = [[None for _ in range(prob.timesteps)] for _ in prob.lasers]

    for t in range(prob.timesteps):
        for i in range(len(prob.lasers)):
            for j in range(len(prob.ufos)):
                if x[i, j, t].value() == 1:
                    targeting[i][t] = j

    return {
        "status": m.status,  # Solver model status
        "remaining": remaining,  # Number of UFOs remaining with health > 0
        "targeting": targeting,  # Targeting info for lasers
    }
