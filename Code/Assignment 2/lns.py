# Copyright subsists in all the material on this repository and vests in the ANU
# (or other parties). Any copying of the material from this site other than for
# the purpose of working on this assignment is a breach of copyright.
#
# In practice, this means that you are not allowed to make any of the existing
# material in this repository nor your assignment available to others (except to
# the course lecturers and tutors) at any time before, during, or after the
# course. In particular, you must keep any clone of this repository and any
# extension of it private at all times.

import random
import time

from nurse import RosteringProblem, read_costs, load_last_roster, save_roster
from pulp import PULP_CBC_CMD

from model import ModelBuilder

# Destroy Method 1
"""
The `destroy_increased_random_shifts` function is implemented to disrupt nurse rosters by selectively removing shifts 
and adjusting the destruction parameters dynamically. This function supports multiple destruction strategies in one 
method, including weighted destruction based on shift type, a gradual increase in destroyed shifts, and larger 
contiguous block removals. Together, these mechanisms give the algorithm flexibility to explore a wide variety of 
roster configurations while balancing randomness with control.

Key Implementation Aspects:
1. **Weighted Random Destruction (First Pass)**:
   - Shift types ('A', 'P', 'N', 'F') are given specific weights to determine which shifts are more likely to be destroyed. 
   - A small randomness factor is added to these weights to vary destruction patterns between iterations, introducing 
     flexibility into which shifts are destroyed.
   - Shifts specified in `avoid_shifts` are excluded, ensuring critical assignments remain undisturbed, which helps 
     maintain the roster’s essential structure.

2. **Multiplicative Destruction Factor (Second Pass)**:
   - In this pass, a defined number of additional small blocks are destroyed, with each block lasting 1-2 days.
   - The `mult_factor` determines the increase in the total number of destroyed shifts beyond the initial rate.
   - This second pass amplifies shift removal selectively, adding significant variation but keeping the damage controlled.

3. **Contiguous Block Destruction (Third Pass)**:
   - After initial and multiplicative destructions, a larger block is destroyed for each nurse, with the length of the 
     block adjusted based on the already-destroyed parts of their schedule. 
   - The function calculates `min_period_length` and `max_period_length` dynamically, preventing excessive roster damage.
   - This step diversifies the solution space by enabling more substantial roster changes, helping escape repetitive patterns.

Together, these destruction phases offer a balanced approach to exploring the search space, where each pass 
complements the others. In the main `large_neighbourhood_search` (LNS) function, parameters such as destruction rate 
and block length can be adjusted when specific thresholds (like cost values) are reached, letting the algorithm 
intensify destruction only when needed. Overall, `destroy_increased_random_shifts` gives fine-grained control 
over the destructiveness applied to the roster.
"""


def destroy_increased_random_shifts(roster, destruction_rate=0.15, min_period_length=8, max_period_length=11,
                                    mult_factor=1,
                                    avoid_shifts=None, shift_weights=None):
    """
    Enhanced destruction function that removes a portion of the roster by destroying shift assignments randomly,
    allowing control over destruction rate, shift types to avoid, weighted destruction, and periodic block destruction.

    Args:
    - roster (list of str): The current roster of all nurses.
    - destruction_rate (float): Percentage of total shifts to destroy.
    - min_period_length (int): Minimum length of block destruction during the second pass.
    - max_period_length (int): Maximum length of block destruction during the second pass.
    - mult_factor (float): Multiplicative factor to increase the count of shifts destroyed.
    - avoid_shifts (set of str, optional): Shift types to avoid destroying, such as 'F' for off-duty.
    - shift_weights (dict of str:float, optional): Destruction weights for each shift type. If not provided, defaults are applied.

    Returns:
    - new_roster (list of str): The updated roster with destroyed shifts.
    - destroyed_indices (list of tuples): The list of destroyed (nurse_idx, day_idx) indices.
    """
    new_roster = roster.copy()
    num_nurses = len(new_roster)  # Total number of nurses in the roster
    num_days = len(new_roster[0])  # Number of days in each nurse's schedule

    # Set default avoid shifts if none provided
    avoid_shifts = avoid_shifts or set()  # No shifts avoided by default

    # If shift_weights not specified, dynamically assign default weights based on a random adjustment
    val = random.random()

    if val > 0.7:
        val *= -1

    # Assign weights to shift types for probability-based destruction, or apply provided shift weights
    shift_weights = shift_weights or {'A': 4 - val, 'P': 3 + val, 'N': 2 + val, 'F': 3 + val}

    # Normalize weights to sum to 1, creating a destruction probability for each shift type
    total_weight = sum(shift_weights.values())
    normalized_weights = {shift: weight / total_weight for shift, weight in shift_weights.items()}

    # Determine the number of shifts to destroy based on the destruction rate
    total_shifts = num_nurses * num_days
    shifts_to_destroy = int(total_shifts * destruction_rate)

    destroyed_indices = set()  # Set for destroyed shifts, improving look-up speed and preventing duplicates

    # First destruction pass: Destroy shifts based on weighted probabilities to vary shift patterns
    while len(destroyed_indices) < shifts_to_destroy:
        # Randomly select a nurse and day to potentially destroy
        nurse_idx = random.randint(0, num_nurses - 1)
        day_idx = random.randint(0, num_days - 1)
        shift = new_roster[nurse_idx][day_idx]  # Get the shift type at this position

        # Check if the selected shift type is avoidable or has already been destroyed
        if (nurse_idx, day_idx) not in destroyed_indices and shift not in avoid_shifts:

            # Destroy with a probability based on normalized weights for the shift type
            if random.random() < normalized_weights.get(shift, 1):
                # Destroy the shift i.e. replace with '.' to indicate destruction
                new_roster[nurse_idx] = new_roster[nurse_idx][:day_idx] + '.' + new_roster[nurse_idx][day_idx + 1:]
                destroyed_indices.add((nurse_idx, day_idx))  # Add to destroyed set for future reference

    # Additional destruction count based on multiplicative factor
    destroyed = 0
    count_shifts_to_destroy = shifts_to_destroy * mult_factor

    # Second destruction pass: Randomly destroy additional short blocks of shifts based on mult_factor
    while destroyed < count_shifts_to_destroy:
        # Randomly select a nurse and day to destroy a small consecutive block of shifts
        nurse_idx = random.randint(0, num_nurses - 1)
        day_idx = random.randint(0, num_days - 1)
        consecutive_days = random.randint(1, 2)  # Length of destruction block (1-2 days)
        end_day_idx = min(day_idx + consecutive_days, num_days)  # Limit destruction to roster length

        # Destroy the small block of shifts for the selected nurse
        for d in range(day_idx, end_day_idx):
            if (nurse_idx, d) not in destroyed_indices:
                new_roster[nurse_idx] = new_roster[nurse_idx][:d] + '.' + new_roster[nurse_idx][d + 1:]
                destroyed_indices.add((nurse_idx, d))
                destroyed += 1  # Increment the count for destroyed shifts

        # Stop if we've reached the required number of destroyed shifts
        if destroyed >= count_shifts_to_destroy:
            break

    # Third destruction pass: Apply contiguous block destruction for each nurse
    for nurse_idx in range(num_nurses):

        # Calculate already destroyed shifts for the nurse, used to adjust block length dynamically
        destroyed_count_for_nurse = sum(1 for day in range(num_days) if (nurse_idx, day) in destroyed_indices)

        # Adjust block destruction range based on previously destroyed shifts
        min_period = min(min_period_length, max(0, max_period_length - destroyed_count_for_nurse - 3))
        max_period = max(min_period, min(max_period_length - destroyed_count_for_nurse, min_period_length + 3))

        # Select a random period length within calculated range
        if min_period <= max_period:
            period_length = random.randint(min_period, max_period)
        else:
            # Fall back to a default or handle cases where the range is invalid
            period_length = min_period_length  # or max_period_length

        # Destroy shifts in the selected block if period length is valid
        if period_length > 0:
            start_day = random.randint(0, num_days - period_length)  # Random starting point for destruction

            # Destroy the block of shifts for the selected nurse
            for day_idx in range(start_day, start_day + period_length):
                if (nurse_idx, day_idx) not in destroyed_indices:
                    new_roster[nurse_idx] = new_roster[nurse_idx][:day_idx] + '.' + new_roster[nurse_idx][day_idx + 1:]
                    destroyed_indices.add((nurse_idx, day_idx))

    return new_roster, list(destroyed_indices)


# Destroy Method 2
"""
The `destroy_consecutive_work_off_blocks_with_dict` function was implemented as a strategic
approach to disrupt the nurse roster by selectively destroying blocks of consecutive work shifts
(A, P, N) followed by off-duty (F) days for each nurse. This design introduces variability into
the search space by targeting specific patterns, thereby helping the algorithm escape local minima.
The method maintains certain scheduling structures while allowing alternative solutions to emerge.

Key Implementation Aspects:
1. **Use of Dictionary Conversion**: To enhance flexibility, this function utilizes `convert_string_to_dict` to 
   transform the string-based roster into a dictionary format, allowing precise targeting and manipulation of specific 
   blocks. After modifications, it uses `convert_dict_to_string` to revert the schedule back to a usable format.

2. **Selective Block Targeting**: The function specifically destroys work-off sequences (work blocks
   immediately followed by off-duty blocks) instead of randomly removing shifts. This maintains the 
   schedule's natural structure and provides controlled variability in the optimization process.

3. **Controlled Destruction Limits**: Minimum and maximum block lengths are defined to avoid 
   excessive disruption and maintain feasible rosters. The block length also adjusts dynamically 
   based on the already destroyed sections, ensuring the method remains balanced.

4. **One-time Targeted Destruction**: To avoid over-destruction, only one work-off block combination 
   is processed per nurse, providing structured change to the schedule and minimizing repair complexity.

5. **Random Additional Block Destruction**: An additional pass is included to introduce minor 
   random destruction, further diversifying the solution without overwhelming randomness.

This function provides a practical balance between roster flexibility and structure preservation, 
ensuring effective exploration while keeping repairs manageable.
"""


def destroy_consecutive_work_off_blocks_with_dict(roster):
    """
    Destroy method: Randomly destroys one consecutive block of workdays (A, P, N) and the following consecutive
    block of off-duty days (F) for each nurse, but only if the off-duty block follows the work block.

    Args:
    - roster (list of str): The current roster of all nurses.

    Returns:
    - new_roster (list of str): The updated roster with destroyed shifts.
    - destroyed_indices (list of tuples): The list of destroyed (nurse_idx, day_idx) indices.
    """
    new_roster = []  # Stores the modified rosters for all nurses
    destroyed_indices = set()  # Track all destroyed indices (nurse, day) for easy lookup

    num_nurses = len(roster)  # Total number of nurses
    num_days = len(roster[0])  # Number of days in each nurse's schedule
    start_day = 0
    last_day = num_days  # Keeps track of the last modified day for spacing destroyed sections

    for nurse_idx, nurse_schedule in enumerate(roster):
        total_consecutive_length = 0  # Tracks the total length of destroyed blocks for each nurse

        # Convert the nurse's string schedule to a dictionary, breaking it into blocks of consecutive shifts
        schedule_dict = convert_string_to_dict(nurse_schedule)

        # Identify all work blocks (A, P, N shifts only) in the schedule
        work_blocks = [(start, block) for start, block in schedule_dict.items() if block['shift'] in ['A', 'P', 'N']]

        new_schedule = convert_dict_to_string(
            schedule_dict)  # Convert dictionary back to a string format for easy updating
        j = 0  # To ensure only one work-off block pair is selected
        while j < 1:
            if work_blocks:
                # Randomly select a work block from available options
                start_day, work_block = random.choice(work_blocks)

                # Check if an off-duty block follows the selected work block
                next_block_start = start_day + work_block['length']
                if next_block_start in schedule_dict and schedule_dict[next_block_start]['shift'] == 'F':
                    off_block = schedule_dict[next_block_start]

                    # Destroy the shifts in the work block
                    for day in range(start_day, start_day + work_block['length']):
                        new_schedule = new_schedule[:day] + '.' + new_schedule[day + 1:]
                        destroyed_indices.add((nurse_idx, day))
                    total_consecutive_length += work_block['length']

                    # Destroy the shifts in the off-duty block
                    for day in range(next_block_start, next_block_start + off_block['length']):
                        new_schedule = new_schedule[:day] + '.' + new_schedule[day + 1:]
                        destroyed_indices.add((nurse_idx, day))
                    total_consecutive_length += off_block['length']

                    last_day = start_day + total_consecutive_length

                    j += 1  # Only one work-off block pair per nurse is processed
                else:
                    # If no valid work-off block combination is found, continue searching
                    continue
            nurse_schedule = new_schedule
        new_roster.append(nurse_schedule)

        # Randomly destroy an additional block of shifts outside the selected work-off block
        period_length = int(random.randint(12 - total_consecutive_length, 14 - total_consecutive_length) / 2)

        # Assign end and beg day based on period length of the previous destruction
        beg_day = random.randint(0, num_days - period_length)
        end_day = random.randint(beg_day + 1, num_days)
        while beg_day == start_day:
            beg_day = random.randint(0, num_days - period_length)
        while end_day == last_day:
            end_day = random.randint(beg_day + 1, num_days)

        # Destroy shifts in the new block for the selected start and end days
        for day_idx in range(beg_day, min(beg_day + period_length, num_days)):
            if (nurse_idx, day_idx) not in destroyed_indices:
                new_roster[nurse_idx] = new_roster[nurse_idx][:day_idx] + '.' + new_roster[nurse_idx][day_idx + 1:]
                destroyed_indices.add((nurse_idx, day_idx))

        for day_idx in range(end_day, min(end_day + period_length, num_days)):
            if (nurse_idx, day_idx) not in destroyed_indices:
                new_roster[nurse_idx] = new_roster[nurse_idx][:day_idx] + '.' + new_roster[nurse_idx][day_idx + 1:]
                destroyed_indices.add((nurse_idx, day_idx))

    return new_roster, list(destroyed_indices)


# Destroy Method 3
"""
The `destroy_every_few_days_with_comparison` function introduces targeted variability into the roster by 
destroying a specified range of days after every block of consecutive workdays. Unlike random destruction, 
this method uses `compared_indices`—the indices identified from differences between the best and second-best 
solutions in the previous iteration of the Large Neighbourhood Search (LNS)—to avoid or prioritize certain indices, 
bringing more informed randomness to the process.

Key Implementation Aspects:
1. **Comparison-Based Destruction**: `compared_indices`, which stores indices where the best and second-best 
   solutions differ, guides the destruction process. This approach maintains promising parts of the roster 
   while exploring changes in sections with potential for improvement.

2. **Controlled Skip and Destroy Blocks**: The function skips controlled amount of consecutive workdays (randomized with 
   `rand_min` and `rand_max`) before destroying a specified range (defined by `min_num` and `max_num`). This adds balance 
   by alternating between workdays and destruction intervals.

3. **Additional Randomized Destruction**: To introduce further diversity, the function performs an additional 
   block destruction on a few randomly selected days. This adds robustness to the search space exploration 
   while maintaining a manageable repair complexity.

This function helps the algorithm focus on key areas for improvement while preventing excessive disruption, 
making the repair phase more efficient.
"""


def destroy_every_few_days_with_comparison(roster, compared_indices=None, min_num=1, max_num=2, rand_min=0, rand_max=2):
    """
    Destroy method that destroys 1-2 days after every 3-4 consecutive workdays.

    Args:
    - roster (list of str): The current roster of all nurses.
    - compared_indices (list of tuples, optional): Indices where destruction should not occur.
                                                   Default is None.
    - min_num (int): Minimum consecutive days to destroy after each interval.
    - max_num (int): Maximum consecutive days to destroy after each interval.
    - rand_min (int): Minimum number of days to skip before destruction.
    - rand_max (int): Maximum number of days to skip before destruction.

    Returns:
    - new_roster (list of str): The updated roster with destroyed shifts.
    - destroyed_indices (list of tuples): The list of destroyed (nurse_idx, day_idx) indices.
    """
    new_roster = roster.copy()  # Copy the original roster to modify
    num_nurses = len(new_roster)  # Number of nurses in the roster
    num_days = len(new_roster[0])  # Number of days in the roster

    destroyed_indices = set()  # Track which indices are destroyed for efficient lookups
    val = random.random()

    # Randomly decide if compared_indices will be considered
    if val > 0.5:
        compared_indices = None

    # If compared_indices is provided, shuffle and subset it to randomize the destruction process
    if compared_indices:
        random.shuffle(compared_indices)

        random_fraction = random.uniform(0.3, 0.7)
        required_length = int(len(compared_indices) * random_fraction)

        compared_indices = sorted(compared_indices[:required_length])

    # Iterate through each nurse to apply destruction on their schedule
    for nurse_idx in range(num_nurses):
        day_idx = 0

        while day_idx < num_days:
            # Determine the number of workdays to skip before applying destruction
            skip_days = random.randint(rand_min, rand_max)

            day_idx += skip_days  # Move forward by the skip_days

            # Destroy a random number (1 or 2) of consecutive days, if within bounds
            destroy_days = random.randint(min_num, max_num)
            for d in range(day_idx, min(day_idx + destroy_days, num_days)):
                if (nurse_idx, d) not in destroyed_indices:
                    # Skip destruction if the day is in compared_indices
                    if compared_indices and (nurse_idx, d) in compared_indices:
                        continue  # Skip destruction for these indices
                    new_roster[nurse_idx] = new_roster[nurse_idx][:d] + '.' + new_roster[nurse_idx][d + 1:]
                    destroyed_indices.add((nurse_idx, d))  # Track destroyed indices

            # Move day index forward by the number of days destroyed to continue the loop
            day_idx += destroy_days

        # Additional randomized destruction: destroy a small block of days at a random starting point
        period_length = random.randint(1, 3)  # Length of the additional destruction block
        start_day = random.randint(0, num_days - period_length)  # Starting day for additional destruction

        # Destroy the small block of consecutive days if they aren't already destroyed
        for day_idx in range(start_day, start_day + period_length):
            if (nurse_idx, day_idx) not in destroyed_indices:
                new_roster[nurse_idx] = new_roster[nurse_idx][:day_idx] + '.' + new_roster[nurse_idx][day_idx + 1:]
                destroyed_indices.add((nurse_idx, day_idx))  # Track destroyed indices

    return new_roster, list(destroyed_indices)


def convert_string_to_dict(schedule_string):
    """
    Converts a string-based schedule to a dictionary where each index corresponds to
    the start of a block of shifts, and the block's shift type and length are stored.

    Args:
    - schedule_string (str): A string of shifts like 'NPPFAAAAFFNNPPAAAFFFN'.

    Returns:
    - schedule_dict (dict): A dictionary where keys are day indices and values are dictionaries
                            containing 'shift' and 'length' of consecutive shifts.
    """
    schedule_dict = {}
    current_shift = schedule_string[0]
    current_start = 0
    current_length = 1

    for i in range(1, len(schedule_string)):
        if schedule_string[i] == current_shift:
            current_length += 1
        else:
            # Save the block when shift changes
            schedule_dict[current_start] = {"shift": current_shift, "length": current_length}
            # Reset for the new shift
            current_shift = schedule_string[i]
            current_start = i
            current_length = 1

    # Add the final block
    schedule_dict[current_start] = {"shift": current_shift, "length": current_length}

    return schedule_dict


def convert_dict_to_string(schedule_dict):
    """
    Converts the dictionary-based schedule back into a string format.

    Args:
    - schedule_dict (dict): A dictionary representing the schedule, where each key is the
                            start day and value contains shift type and block length.

    Returns:
    - schedule_string (str): The schedule as a string of shifts like 'NPPFAAAAFFNNPPAAAFFFN'.
    """
    schedule_string = [''] * 21  # Assuming we are working with a 21-day schedule

    for day, block in schedule_dict.items():
        shift_type = block['shift']
        block_length = block['length']
        # Fill the schedule string with the shift type repeated for the block's length
        schedule_string[day:day + block_length] = [shift_type] * block_length

    return ''.join(schedule_string)


def repair_solution(prob, destroyed_roster, destroyed_indices, costs):
    """
    Repairs the partially destroyed roster by using the ModelBuilder to reconstruct missing parts.
    This function attempts to maintain feasible shift assignments by preserving non-destroyed elements
    while re-optimizing destroyed sections to achieve a low-cost, feasible solution.

    Args:
    - prob (RosteringProblem): The rostering problem instance with constraints and parameters.
    - destroyed_roster (list of str): The roster with certain shifts removed (indicated by '.').
    - destroyed_indices (list of tuples): Indices representing the positions of destroyed ('.') shifts.
    - costs (dict): Cost factors associated with the problem, such as shift preferences and penalties.

    Returns:
    - repaired_solution (list of str): The repaired roster with all destroyed shifts re-assigned
      while preserving non-destroyed parts, or None if no solution is found.
    """
    # Initialize a ModelBuilder to construct and solve the repair model
    mb = ModelBuilder(prob)
    model = mb.build_model(costs)  # Build the initial model using given costs

    # Lock non-destroyed parts of the roster by setting them as constraints in the model
    for nurse_idx in range(len(destroyed_roster)):
        for day_idx in range(len(destroyed_roster[nurse_idx])):
            shift = destroyed_roster[nurse_idx][day_idx]

            # Only add constraints for non-destroyed shifts
            if shift != '.':
                model += mb._choices[nurse_idx][day_idx][shift] == 1  # Fix shift in model

    # Attempt to solve the model to fill in destroyed ('.') shifts
    res = model.solve(PULP_CBC_CMD(msg=False))

    # Check the result of the solve operation
    if res != 1:
        # Return None if no feasible solution was found
        return None

        # Extract the full solution with repaired shifts
    repaired_solution = mb.extract_solution()

    return repaired_solution


def compare_solutions(best_solution, second_best_solution):
    """
    Compares two solutions and returns a list of indices where the solutions differ.

    Args:
    - best_solution (list of str): The best solution's roster (list of nurse schedules as strings).
    - second_best_solution (list of str): The second-best solution's roster (list of nurse schedules as strings).

    Returns:
    - compared_indices (list of tuples): List of (nurse_idx, day_idx) pairs where the two solutions differ.
    """
    compared_indices = []

    # Iterate through each nurse's schedule
    for nurse_idx in range(len(best_solution)):
        best_nurse_schedule = best_solution[nurse_idx]
        second_best_nurse_schedule = second_best_solution[nurse_idx]

        # Iterate through each day in the nurse's schedule
        for day_idx in range(len(best_nurse_schedule)):
            if best_nurse_schedule[day_idx] != second_best_nurse_schedule[day_idx]:
                compared_indices.append((nurse_idx, day_idx))

    return compared_indices


def large_neighbourhood_search(prob, costs, initial_solution, destroy_method, max_iterations=1000,
                               destruction_rate=0.2):
    """
    Perform Large Neighbourhood Search (LNS) to optimize the nurse rostering problem using a specified destroy method.

    Args:
    - prob (RosteringProblem): Instance of the rostering problem containing constraints and parameters.
    - costs (dict): Cost factors related to shift preferences, coverage requirements, and workload balancing.
    - initial_solution (list of str): The initial feasible roster, represented as a list of nurse schedules.
    - destroy_method (function): The destruction function used to partially 'destroy' parts of the roster.
    - max_iterations (int): The maximum number of iterations to perform (default: 1000).
    - destruction_rate (float): Percentage of the roster to destroy each iteration (default: 0.2).

    Returns:
    - best_solution (list of str): The optimized roster with the lowest recorded cost.
    - best_cost (float): The cost associated with the best solution found.
    """
    # Initialize best solution with initial values
    best_solution = initial_solution
    best_cost = prob.cost(best_solution, costs)
    print(best_cost)

    # Temporary values for adjusting the destruction parameters dynamically during iterations
    temp = 0
    mult_factor = 1
    min_period_length = 3
    max_period_length = 10
    compared_indices = None  # Tracks positions with differing values between best solutions

    for iteration in range(max_iterations):

        # Apply the chosen destroy method to create a partially destroyed roster
        # and track the indices of destroyed elements

        if destroy_method == destroy_consecutive_work_off_blocks_with_dict:
            # Destroy consecutive work and off blocks with specific logic
            destroyed_roster, destroyed_indices = destroy_method(best_solution)

        elif destroy_method == destroy_increased_random_shifts:
            # Apply general destruction based on given rate, period length, and multiplier
            destroyed_roster, destroyed_indices = destroy_method(best_solution, destruction_rate,
                                                                 min_period_length, max_period_length, mult_factor)

            # Adjust destruction parameters dynamically if cost threshold met
            if best_cost < 89:
                # Gradually increase destruction rate if below max limit
                if destruction_rate < 0.28:
                    destruction_rate += 0.00025
                # Reduce block length periodically for more refined destruction
                if temp > 35:
                    min_period_length = max(0, min_period_length - 1)
                    max_period_length = max(0, max_period_length - 1)
                    temp = 0  # Reset temp counter after adjustment
                temp += 1
                mult_factor = 4 / 3  # Increase multiplier factor for larger destructions
                #print(destroyed_roster)

        elif destroy_method == destroy_every_few_days_with_comparison:
            # Perform selective destruction while comparing with a prior solution for more variability
            destroyed_roster, destroyed_indices = destroy_method(best_solution, compared_indices)

        # Repair the solution by filling in destroyed parts with feasible values
        repaired_solution = repair_solution(prob, destroyed_roster, destroyed_indices, costs)
        if repaired_solution is None:
            repaired_solution = best_solution  # Fall back to current best if repair fails
        repaired_cost = prob.cost(repaired_solution, costs)  # Calculate the cost of the repaired solution

        # Update the best solution if an improvement is found
        if repaired_cost < best_cost:
            # Retain prior best solution for comparison
            second_best_solution = best_solution
            best_solution = repaired_solution  # Update to new best solution
            best_cost = repaired_cost  # Update best cost
            print(best_cost)

            # Track differing indices between best solutions for future comparison-based destruction
            compared_indices = compare_solutions(best_solution, second_best_solution)

            # Save the updated best solution to a file
            save_roster(best_solution)

    return best_solution, best_cost  # Return optimized solution and its cost


if __name__ == '__main__':
    prob = RosteringProblem()
    costs = read_costs(prob)
    initial_solution = load_last_roster()
    initial_solution = [line.strip() for line in initial_solution]

    import sys

    # Destroy Methods
    destroy_methods = [destroy_consecutive_work_off_blocks_with_dict, destroy_every_few_days_with_comparison,
                       destroy_increased_random_shifts]

    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) not in [1, 2, 3]:
        sys.exit(1)

    index = int(sys.argv[1]) - 1
    max_iterations = 500
    if index == 0:
        max_iterations = 350
    elif index == 1:
        max_iterations = 300

    best_solution, best_cost = large_neighbourhood_search(prob, costs, initial_solution, destroy_methods[index],
                                                          max_iterations)

    print(f"LNS completed. Best solution cost: {best_cost}")
    save_roster(best_solution)
