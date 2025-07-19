# solution2.py
# Author: Saksham Gupta (UID: 7726995)
#
# Purpose:
# This module is designed to generate, validate, and optimize feasible nurse rosters by
# handling individual scheduling constraints and ensuring group compliance with Constraint C7.
# Individual schedules are created for each nurse, checked for feasibility for all constraints besides C7.
# After generating these schedules, the module uses a group-level validation to confirm the
# minimum staffing levels needed per shift across all days. The automated process
# involves generating random combinations, validating each schedule, and selecting an
# optimized roster set that meets both individual and group-level requirements.

from typing import List, Set, Dict
from collections import defaultdict
import solution1 as temp
from nurse import RosteringProblem, SHIFT_MORNING, SHIFT_AFTERNOON, SHIFT_NIGHT, SHIFT_OFFDUTY, DAYS_PER_WEEK, SHIFTS

from solution1 import (nb_nurses, nb_weeks, min_nb_consecutive_days, max_nb_consecutive_days,
                       nb_off_duty_days, min_nb_consecutive_work_days, max_nb_consecutive_work_days,
                       min_nb_consecutive_off_days, max_nb_consecutive_off_days, shift_requirements)

from solution1 import consecutive_numbers


def feasible_c7(roster: List[str]) -> str:
    """
        Checks if the given nurse roster meets Constraint C7, which requires a minimum number
        of nurses for each shift (morning, afternoon, night) across all days. If the roster
        does not meet the required number of nurses for any shift on a particular day,
        an error message is returned. Otherwise, returns None, indicating the roster is feasible.

        Args:
        - roster (List[str]): The list of nurse schedules, where each string represents a
          nurse's schedule over all days.

        Returns:
        - str: An error message if the roster does not meet Constraint C7 requirements; otherwise, None.
    """
    n_nurses = nb_nurses

    try:
        # Iterate over each shift type to ensure the minimum required nurses are present each day
        for shift_type, min_nb in shift_requirements.items():
            # Collect nurses assigned to the current shift type on day d
            for d in range(nb_weeks * DAYS_PER_WEEK):
                nurses = [i for i in range(n_nurses) if roster[i][d] == shift_type]
                # Check if the number of nurses meets the minimum required for the shift
                if len(nurses) < min_nb:
                    return f'Not enough nurses on shift {shift_type} for day {d}'
    except Exception as e:
        return f'Error occured at feasible C7'

    return None


def feasibility(nurse: str) -> str:
    '''
    Checks if the provided nurse's schedule satisfies individual feasibility constraints i.e. C1-C6 & C8

    Borrowed from nurse.py and repurposed for this file.

    Args:
    - nurse (str): The nurse's 21-day schedule string.

    Returns:
    - str: Describes the first constraint violation encountered. Returns None if
           all constraints are satisfied, indicating a feasible schedule.
    '''
    # Constraint C0: contains the right number of rows and columns.

    if len(nurse) != nb_weeks * DAYS_PER_WEEK:
        return f'Wrong shift-length for nurse ({len(nurse)})'

    for d in range(nb_weeks * DAYS_PER_WEEK):
        shift = nurse[d]
        if shift not in SHIFTS:
            return f'Wrong type of shift for nurse on day {d} ({shift})'

    # Constraint C2: 6 off-duty days per nurse
    nb_off_duty_days = len([d for d in range(nb_weeks * DAYS_PER_WEEK) if nurse[d] == SHIFT_OFFDUTY])
    if nb_off_duty_days != nb_off_duty_days:
        return f'Wrong number of off-duty days for nurse({nb_off_duty_days})'

    # Constraint C3: number of consecutive days in a given shift
    for shift_type in {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT}:
        consecutives = consecutive_numbers(nurse, nb_weeks * DAYS_PER_WEEK, shift_type)
        for k, v in consecutives.items():
            if v < min_nb_consecutive_days or v > max_nb_consecutive_days:
                return f'Wrong number of consecutive days for shift {shift_type} of nurse starting from day {k}'

    # Constraint C4: Consecutive number of work days for a nurse
    consecutives = consecutive_numbers(nurse, nb_weeks * DAYS_PER_WEEK,
                                       {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT})
    for k, v in consecutives.items():
        if v < min_nb_consecutive_work_days or v > max_nb_consecutive_work_days:
            return f'Wrong number of consecutive work days for shift of nurse starting from day {k}'

    # Constraint C5: Consecutive number of off-duty days for a nurse
    consecutives = consecutive_numbers(nurse, nb_weeks * DAYS_PER_WEEK, {SHIFT_OFFDUTY})
    for k, v in consecutives.items():
        if v < min_nb_consecutive_off_days or v > max_nb_consecutive_off_days:
            return f'Wrong number of consecutive off-duty days for shift of nurse starting from day {k}'

    # Constraint C6: at least one week-end off-duty day
    weekenddays = (
            [5 + (DAYS_PER_WEEK * w) for w in range(nb_weeks)]  # Saturdays
            + [6 + (DAYS_PER_WEEK * w) for w in range(nb_weeks)]  # Sundays
    )
    weekendshifts = {nurse[d] for d in weekenddays}
    if SHIFT_OFFDUTY not in weekendshifts:
        return f'Nurse does not have an off-duty day on week-ends'

    # Constraint C8: order of shift (morning -> night -> afternoon)
    previous_work_shift = None
    across_offduty = False
    for d in range(2 * nb_weeks * DAYS_PER_WEEK):
        shift = nurse[d % (nb_weeks * DAYS_PER_WEEK)]
        if shift == SHIFT_OFFDUTY:
            across_offduty = True
            continue
        if shift == SHIFT_MORNING and (
                previous_work_shift == SHIFT_NIGHT
                or (across_offduty and previous_work_shift == SHIFT_MORNING)
        ):
            return f"Wrong shift order for nurse on day {d} ({previous_work_shift} -> {shift})"
        if shift == SHIFT_NIGHT and (
                previous_work_shift == SHIFT_AFTERNOON
                or (across_offduty and previous_work_shift == SHIFT_NIGHT)
        ):
            return f"Wrong shift order for nurse on day {d} ({previous_work_shift} -> {shift})"
        if shift == SHIFT_AFTERNOON and (
                previous_work_shift == SHIFT_MORNING
                or (across_offduty and previous_work_shift == SHIFT_AFTERNOON)
        ):
            return f"Wrong shift order for nurse on day {d} ({previous_work_shift} -> {shift})"
        previous_work_shift = shift
        across_offduty = False

    return None


def generate_combination(prob: RosteringProblem):
    """
    Generates a nurse's complete schedule by constructing and combining work shift blocks
    with off-duty days, adhering to specified scheduling constraints. This function
    leverages methods from Solution 1 to ensure that individual schedules meet high feasibility
    standards before they are grouped for final roster validation.

    Args:
    - prob (RosteringProblem): The rostering problem object.

    Returns:
    - full_schedule: A full schedule that includes work and off-duty blocks.
    """
    # Generate work shift blocks with constraints
    work_blocks = temp.generate_work_blocks()

    # Insert necessary off-duty blocks into the work shift blocks, enforcing additional constraints
    full_schedule = temp.insert_off_duty_blocks(work_blocks)

    return full_schedule


def flatten_schedule(schedule):
    """
        Flattens a nested list-based schedule into a single string representation.

        Args:
        - schedule (list of lists): A nurse's schedule represented as a list of blocks,
                                    where each block is a list of consecutive shifts
                                    (e.g., [['A', 'A'], ['F', 'F'], ['N', 'N', 'N']]).

        Returns:
        - flattened (str): A single string where each character represents a shift for one day,
                           with all blocks joined sequentially (e.g., "AAFFNNN").
    """
    flattened = ''.join([''.join(block) for block in schedule])
    return flattened


def greedy_assign_nurses(valid_nurses, nb_days=21, shift_requirements=None):
    """
    Greedy assignment to ensure the roster meets the minimum shift requirements for each day.
    Nurses that fill the largest gaps are assigned first, with weighted contributions for under-covered shifts.

    Args:
    - valid_nurses (List[str]): List of nurse schedules (each string represents a schedule of 21 days).
    - nb_days (int): Number of days to check (default is 21).
    - shift_requirements (dict): Minimum number of nurses required per shift.

    Returns:
    - selected_nurses (List[str]): A list of nurses that meet the shift requirements.
    - shift_counts_per_day (List[Dict[str, int]]): Shift coverage per day after assignment.
    - nurse_contributions (List[Tuple[str, int]]): List of nurses and their contributions.
    """
    if shift_requirements is None:
        shift_requirements = {
            'A': 4,  # Morning shift
            'P': 3,  # Afternoon shift
            'N': 2,  # Night shift
        }

    selected_nurses = []  # Final list of selected nurses
    nurse_contributions = []  # Track nurse contributions
    shift_counts_per_day = [defaultdict(int) for _ in range(nb_days)]  # Track shift coverage per day
    shifts_needed = [shift_requirements.copy() for _ in range(nb_days)]  # Track which shifts still need more nurses

    def calculate_weighted_contribution(nurse_schedule):
        """
        Internal function to calculate the weighted contribution of a nurse based on the extent to which each shift is under-covered.
        This function gives higher priority to shifts that need more nurses, making sure the most critical gaps are filled first.

        Args:
        - nurse_schedule (str): The schedule of a single nurse, represented as a string of shifts for each day.

        Returns:
        - contribution (int): A weighted score indicating how much this nurse's schedule will contribute
                          to meeting the required staffing levels.
        """
        contribution = 0
        for day in range(nb_days):
            shift = nurse_schedule[day]
            # Only consider shifts that are under-covered and require more nurses
            if shift in shift_requirements and shifts_needed[day][shift] > 0:
                # Weight the contribution higher for shifts that are more under-covered
                weight = shift_requirements[shift] - shift_counts_per_day[day][shift]
                contribution += weight  # Add weighted contribution for this shift
        return contribution

    while True:
        best_nurse = None
        best_contribution = 0

        # Iterate through valid nurses to find the one that fills the most needed shifts, with weighted contributions
        for nurse_schedule in valid_nurses:
            contribution = calculate_weighted_contribution(nurse_schedule)

            if contribution > best_contribution:  # Select the nurse with the highest weighted contribution
                best_nurse = nurse_schedule
                best_contribution = contribution

        # Exit if no nurse contributes to filling gaps
        if best_nurse is None or best_contribution == 0:
            break

        # Add the best nurse to the roster, record their contribution, and update shift counts
        selected_nurses.append(best_nurse)
        nurse_contributions.append((best_nurse, best_contribution))
        for day in range(nb_days):
            shift = best_nurse[day]
            if shift in shift_requirements and shifts_needed[day][shift] > 0:
                shift_counts_per_day[day][shift] += 1
                shifts_needed[day][shift] -= 1

        valid_nurses.remove(best_nurse)  # Remove the selected nurse from the pool

        # Exit if all shifts are fully staffed
        if all(shift_needed <= 0 for day in shifts_needed for shift_needed in day.values()):
            break

    # Sort nurses by their contribution in descending order
    nurse_contributions.sort(key=lambda x: x[1], reverse=True)
    sorted_nurses = [nurse for nurse, contribution in nurse_contributions]

    return sorted_nurses, shift_counts_per_day, nurse_contributions


def validate_and_save(prob=None, num_combinations=100000, check_num_combinations=1000):
    """
    Generates, validates, and stores feasible nurse schedules, combining individual schedules
    into a complete roster that meets all defined constraints, including group constraints (like C7).
    It uses a random sampling approach to explore potential solutions, validating each against the
    required staffing levels and individual scheduling constraints.

    This function coordinates the process by generating individual nurse schedules, checking them
    against individual constraints, and grouping valid schedules.

    Args:
    - prob (RosteringProblem, optional): An instance of RosteringProblem, containing the setup for
                                         scheduling constraints. Defaults to None.
    - num_combinations (int): The maximum number of random schedule combinations to generate.
    - check_num_combinations (int): The number of combinations to check for group validity at once.

    Returns:
    - valid_combinations (List[str]): A list of validated nurse schedules at the minimum count required,
                                    satisfying all 8 constraints.
    - len(valid_combinations) (int): The count of valid schedules found, typically 16 for a complete roster.
    """
    check_combinations = []  # Temporary list to hold schedules that meet individual constraints
    valid_combinations = []  # List to store fully validated nurse schedules that meet all constraints

    # Initialize the rostering problem instance if not provided
    if not prob:
        prob = RosteringProblem()

    j = 0  # Counter to limit the number of check rounds

    for i in range(num_combinations):

        # Step 1: Generate a random individual nurse schedule
        combination = generate_combination(prob)

        # Convert the generated schedule to a flattened roster format for easier validation
        converted_combination = flatten_schedule(combination)

        # Check if the schedule meets all individual nurse constraints
        if feasibility(converted_combination) is None:
            check_combinations.append(converted_combination)

        # Once enough combinations are gathered, validate as a group for shift coverage (C7)
        if len(check_combinations) == check_num_combinations:
            j += 1

            # Use greedy assignment to fulfill minimum shift requirements
            optimized_nurses, shift_counts_per_day, nurse_contributions = greedy_assign_nurses(check_combinations)

            # Validate the group of schedules against the collective staffing constraint C7
            if feasible_c7(optimized_nurses) is None:

                valid_combinations = optimized_nurses

                break  # Found a valid set, exit loop

            else:
                check_combinations = []  # Reset for next batch if C7 is not satisfied
                if j == 5:
                    break # Limit the number of check rounds to avoid excess computation

    return valid_combinations, len(valid_combinations)
