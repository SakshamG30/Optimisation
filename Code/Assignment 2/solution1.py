# solution1.py
# Author: Saksham Gupta (UID: 7726995)
#
# Purpose:
# This file is a helper module responsible for generating individual nurse schedules
# that satisfy all personal scheduling constraints except for Constraint C7, which
# requires validation across multiple nurses to ensure minimum staffing levels per shift.
# It focuses on creating blocks of shifts and off-duty days that align with given
# constraints, such as consecutive shift limits, required off-duty days, and shift ordering.
# These individual schedules will later be validated in groups to meet the collective
# requirements specified by Constraint C7.


import random
from typing import List, Set, Dict

from nurse import RosteringProblem, SHIFT_MORNING, SHIFT_AFTERNOON, SHIFT_NIGHT, SHIFT_OFFDUTY, DAYS_PER_WEEK

remaining_work_day = False
di_remaining_work_day = False

prob = RosteringProblem()

nb_nurses = prob._nb_nurses
nb_weeks = prob._nb_weeks
min_nb_consecutive_days = prob._min_nb_consecutive_days
max_nb_consecutive_days = prob._max_nb_consecutive_days
nb_off_duty_days = prob._nb_off_duty_days
min_nb_consecutive_work_days = prob._min_nb_consecutive_work_days
max_nb_consecutive_work_days = prob._max_nb_consecutive_work_days
min_nb_consecutive_off_days = prob._min_nb_consecutive_off_days
max_nb_consecutive_off_days = prob._max_nb_consecutive_off_days
shift_requirements = prob._shift_requirements

WEEKEND_DAYS = (
        [5 + (DAYS_PER_WEEK * w) for w in range(nb_weeks)]  # Saturdays
        + [6 + (DAYS_PER_WEEK * w) for w in range(nb_weeks)]  # Sundays
)


def generate_work_blocks():
    """
    Generate blocks of workdays (A, P, N) with a total of exactly 15 workdays,
    where each block is between 2 and 4 days long, ensuring that no consecutive
    blocks are of the same shift type and respecting the order A -> N -> P -> A.
    """
    work_blocks = []
    remaining_workdays = 15
    previous_shift = None
    random_min = 3  # Start with a minimum block size of 3 to better handle circular constraints

    # Step 1: Generate blocks of 2-4 shifts, strictly following the order A -> N -> P -> A as required by Constraint 8
    def get_next_shift(previous_shift, preference=None):
        """
        Determine the next shift type based on the previous shift and optional preference.

        Args:
        - previous_shift: The shift type of the last block.
        - preference: Optional shift preference ('A-heavy', 'P-heavy', 'N-heavy').

        Returns:
        - The next shift type, chosen based on order or preference.
        """
        # Set shift probabilities based on any preference (default is balanced)
        if preference == 'A-heavy':
            weights = [0.6, 0.2, 0.2]  # Higher probability for Morning (A)
        elif preference == 'P-heavy':
            weights = [0.2, 0.6, 0.2]  # Higher probability for Afternoon (P)
        elif preference == 'N-heavy':
            weights = [0.2, 0.2, 0.6]  # Higher probability for Night (N)
        else:
            weights = [0.33, 0.33, 0.33]  # Equal probability if no preference

        if previous_shift is None:
            # If no previous shift, start with a random shift but follow the weighted preference
            return random.choices([SHIFT_MORNING, SHIFT_NIGHT, SHIFT_AFTERNOON], weights)[0]

        val = random.random()

        # Enforce strict order A -> N -> P -> A, with a 10% chance to repeat the previous shift
        if val < 0.1 and previous_shift is not None:
            return previous_shift
        # Follow the strict A -> N -> P -> A shift order for Constraint 8
        if previous_shift == SHIFT_MORNING:
            return SHIFT_NIGHT
        elif previous_shift == SHIFT_NIGHT:
            return SHIFT_AFTERNOON
        elif previous_shift == SHIFT_AFTERNOON:
            return SHIFT_MORNING

    # Generate workday blocks, each between 2 to 4 days, until 15 workdays are allocated
    while remaining_workdays > 1:
        # Create a block of 2-4 shifts within the remaining workdays
        block_size = random.randint(random_min, min(4, remaining_workdays))

        # Determine the shift type based on previous shift and strict order A -> N -> P -> A
        shift_type = get_next_shift(previous_shift) if previous_shift else get_next_shift(None)

        work_blocks.append([shift_type] * block_size)
        remaining_workdays -= block_size
        previous_shift = shift_type  # Update to current shift
        random_min = 2  # Set minimum block size to 2 after first block

    last_block_shift = work_blocks[-1][0]
    first_block_shift = work_blocks[0][0]
    last_block_size = len(work_blocks[-1])
    first_block_size = len(work_blocks[0])

    # Step 2: Handle circular consistency where the end must follow shift order A -> N -> P -> A
    if get_next_shift(last_block_shift) != first_block_shift:
        # Adjust first block to maintain circular consistency
        work_blocks[0][0] = get_next_shift(last_block_shift)

    # Step 3: If exactly 1 workday is left, add it and maintain shift order
    if remaining_workdays == 1:
        last_shift = get_next_shift(last_block_shift)  # Follow strict shift order

        # Avoid blocks exceeding 4 days by adjusting the last shift type
        if last_block_size == 4 and last_block_shift == last_shift:
            last_shift = get_next_shift(last_block_shift)

        # Avoid circular conflicts with the first block
        if first_block_size == 4 and first_block_shift == last_shift:
            last_shift = get_next_shift(first_block_shift)

        work_blocks.append([last_shift])  # Add the last workday as its own block

        # Step 4: Adjust the first block's first shift to maintain circular consistency
        if last_shift == first_block_shift and len(work_blocks[0]) > 1:
            work_blocks[0][0] = get_next_shift(last_shift)

    # Step 5: If the first block is too small or has only 1 day, merge it with the next block
    if len(work_blocks[0]) == 1 and len(work_blocks) > 1:
        work_blocks[1] = work_blocks[0] + work_blocks[1]
        work_blocks.pop(0)  # Remove the first block after merging

    return work_blocks


def has_off_duty_on_weekend(schedule):
    """
    Check if the schedule has at least one off-duty day (F) on the weekend.
    Weekend days are defined by indices 5, 6, 12, 13, 19, 20.

    Args:
    - schedule (list of lists): The final schedule (list of lists where each sub-list is a block).

    Returns:
    - bool: True if there is at least one off-duty day on the weekend, False otherwise.
    """
    flat_schedule = [shift for block in schedule for shift in block]  # Flatten the list of lists into a single list

    # Check if any weekend day has an off-duty day (F)
    for day in WEEKEND_DAYS:
        if day < len(flat_schedule) and flat_schedule[day] == SHIFT_OFFDUTY:
            return True
    return False


def find_replaceable_weekend_index(final_schedule):
    """
    Search for a weekend day (5, 6, 12, 13, 19, or 20) that corresponds to the first or last
    index of a work block (i.e., a block with work shifts like A, P, or N) where the block's length
    is greater than 2. This is used to help satisfy Constraint 6, ensuring each nurse has at least
    one off-duty day on a weekend without violating other constraints on work block structure.

    Args:
    - final_schedule (list of lists): The final schedule (list of lists where each sub-list is a block).

    Returns:
    - tuple: (block_idx, weekend_day, block_position) where block_idx is the index of the block,
             weekend_day is the exact weekend day index in the overall schedule, and
             block_position is either 0 (for the first index of the block) or length-1 (for the last index of the block).
    - None: If no such weekend day is found.
    """

    current_position = 0  # To track the overall position in the flattened schedule

    for block_idx, block in enumerate(final_schedule):
        block_len = len(block)

        # Check if the block is a work block (contains work shifts A, P, or N)
        if any(shift in {SHIFT_MORNING, SHIFT_AFTERNOON, SHIFT_NIGHT} for shift in block):
            # Only consider blocks with a length greater than 2
            if block_len > 2:
                # Check if the start or end of the block aligns with a weekend day
                block_start_position = current_position  # Start of the block
                block_end_position = current_position + block_len - 1  # End of the block

                # Iterate through the weekend days
                for weekend_day in WEEKEND_DAYS:

                    # Check if the weekend falls at the start or end of the block
                    if weekend_day == block_start_position:
                        return block_idx, weekend_day, 0  # Replaceable at the start
                    elif weekend_day == block_end_position:
                        return block_idx, weekend_day, block_len - 1  # Replaceable at the end

        # Move to the next block by updating current_position
        current_position += block_len

    return None  # Return None if no valid weekend day is found


def replace_with_off_duty(final_schedule, block_idx, block_position, weekend_position):
    """
    Replace the shift at the specified block index and block position with 'F' (off-duty) to satisfy
    Constraint 6, ensuring each nurse has at least one off-duty day on the weekend. This function
    does so without violating other constraints, such as maintaining off-duty block sizes and
    distribution requirements across the schedule.

    Args:
    - final_schedule (list of lists): The final schedule as a list of lists.
    - block_idx (int): The index of the block where the replacement should occur.
    - block_position (int): The position within the block to replace (either 0 or it's length - 1).
    - weekend_position (int): The weekend index being processed (helps cycle through weekends).

    Returns:
    - final_schedule (list of lists): Updated schedule with the shift replaced by 'F' and moved to maintain balance.
    """

    # Ensure valid block index and block position
    if 0 <= block_idx < len(final_schedule):

        # If the replacement is at the start of the block, we will move the shift to the previous block
        if block_position == 0:
            previous_off_block_idx = block_idx - 1
            # Check if the previous block is an off-duty block and can fit an "F"
            if previous_off_block_idx >= 0 and all(
                    shift == SHIFT_OFFDUTY for shift in final_schedule[previous_off_block_idx]):
                if len(final_schedule[previous_off_block_idx]) < 3:
                    # Add "F" to the previous block
                    final_schedule[previous_off_block_idx].append(SHIFT_OFFDUTY)

                    # Now search for the next block with all Fs to remove one F
                    for next_idx in range(block_idx + 1, len(final_schedule)):
                        if all(shift == SHIFT_OFFDUTY for shift in final_schedule[next_idx]):
                            if SHIFT_OFFDUTY in final_schedule[next_idx]:
                                final_schedule[next_idx].remove(SHIFT_OFFDUTY)
                                break

        # If the replacement is at the end of the block, move it to the next block
        else:
            next_off_block_idx = block_idx + 1
            if next_off_block_idx < len(final_schedule):
                # Find the next block with only Fs and add one "F"
                if all(shift == SHIFT_OFFDUTY for shift in final_schedule[next_off_block_idx]):
                    if len(final_schedule[next_off_block_idx]) < 3:
                        final_schedule[next_off_block_idx].insert(0, SHIFT_OFFDUTY)

                        # Now search for the previous block with all Fs to remove one F
                        for prev_idx in range(block_idx - 1, -1, -1):
                            if all(shift == SHIFT_OFFDUTY for shift in final_schedule[prev_idx]):
                                if SHIFT_OFFDUTY in final_schedule[prev_idx]:
                                    final_schedule[prev_idx].remove(SHIFT_OFFDUTY)
                                    break

    return final_schedule


def check_consecutive_workdays(final_schedule):
    """
    Checks if the nurse's final schedule contains any consecutive workday blocks (A, N, P)
    that are less than 3 or more than 7 days long.
    Handles circular schedules as well.

     Args:
    - final_schedule (list of lists): The schedule for a nurse, represented as a list of lists
                                      where each sub-list is a block of shifts.

    Returns:
    - tuple:
        - bool: True if there is a violation of consecutive workday constraints (less than 3 or more than 7),
                False if the schedule is valid.
        - dict or None: A dictionary detailing the positions and lengths of consecutive blocks
                        if there is a violation, or None if the schedule is valid.
    """

    nurse = ''.join([''.join(block) for block in final_schedule])

    if len(nurse) != 21:
        return False, None

    consecutives = consecutive_numbers(nurse, 21,
                                       {SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT})

    # Check for violations where any consecutive workday block is shorter than 3 or longer than 7 days
    for k, v in consecutives.items():
        if v < 3 or v > 7:
            return True, consecutives
    return False, consecutives


def check_and_adjust_consecutive_workdays(schedule_dict, max_consecutive_days=7):
    """
    CChecks the schedule for consecutive workday blocks that exceed the maximum allowed days
    (default: 7 days) and adjusts by inserting an off-duty block ('F') where possible.
    This function ensures compliance with Constraint 4, which requires consecutive workday
    blocks to stay within a specified limit, while preserving the integrity of other
    scheduling constraints.

    Args:
    - schedule_dict (dict): The dictionary representing the schedule, where each key is the day
                            index, and each value is a dictionary with 'shift' and 'length' details.
    - max_consecutive_days (int): The maximum allowed number of consecutive workdays, default is 7.

    Returns:
    - dict: The adjusted schedule dictionary after ensuring no work blocks exceed max_consecutive_days,
            with inserted off-duty blocks to break up long workday sequences.
    """
    adjusted_schedule = {}
    consecutive_count = 0
    previous_day = None
    off_block_to_use = None  # Track off-duty blocks that can be used

    # First pass: identify any available off-duty (F) blocks for potential use
    for day, block in sorted(schedule_dict.items()):
        shift_type = block['shift']
        if shift_type == 'F':
            off_block_to_use = (day, block)  # Save position and length of the off-duty block

    # Second pass: review and adjust workday blocks if they exceed max_consecutive_days
    for day, block in sorted(schedule_dict.items()):
        shift_type = block['shift']
        block_length = block['length']

        if shift_type != 'F':  # Handle only workday blocks (A, N, P)
            consecutive_count += block_length
            if consecutive_count > max_consecutive_days and off_block_to_use:
                off_day, off_block = off_block_to_use

                # Insert the off-duty block in the current position to split the long work block
                adjusted_schedule[previous_day] = schedule_dict[previous_day]  # Append previous work block
                adjusted_schedule[day] = off_block  # Insert the off-duty block to break the sequence
                adjusted_schedule[day + 1] = block  # Reinsert the current work block after the off-duty block
                consecutive_count = block_length  # Reset count for the new block sequence

                # Remove the off-duty block from its original position
                del schedule_dict[off_day]
                off_block_to_use = None  # Reset off-duty block tracker as it has been used
            else:
                adjusted_schedule[day] = block  # Append work block as no adjustment is needed
        else:
            consecutive_count = 0  # Reset if encountering an off-duty block
            adjusted_schedule[day] = block  # Append off-duty block

        previous_day = day  # Update the previous day tracker for the next iteration

    return adjusted_schedule


def handle_circular_consecutive_workdays(schedule_dict, max_consecutive_days=7):
    """
    CChecks for circular consecutive workdays that exceed the max_consecutive_days limit by
    combining the last and first work blocks, in order to satisfy Constraint 4. If necessary,
    this function inserts an off-duty block ('F') to break long workday sequences that span
    from the end to the start of the schedule, ensuring that the maximum limit on consecutive
    workdays is not exceeded.

    Args:
    - schedule_dict (dict): The dictionary representing the schedule, where each key is a day
                            index and each value is a dictionary with 'shift' and 'length' details.
    - max_consecutive_days (int): The maximum allowed number of consecutive workdays.

    Returns:
    - dict: The adjusted schedule dictionary with off-duty blocks inserted if necessary to
            comply with consecutive workday constraints in a circular schedule.
    """
    adjusted_schedule = schedule_dict.copy()
    consecutive_count = 0  # Track total consecutive workdays
    off_block_to_use = None  # Track off-duty blocks that can be used

    # Identify any available off-duty ('F') block that can be used to break long workday sequences
    for day, block in sorted(adjusted_schedule.items()):
        shift_type = block['shift']
        if shift_type == 'F':
            off_block_to_use = (day, block)  # Save the off-duty block's position

    # Initialize variables for the last and first work blocks and their positions
    last_work_block = None
    first_work_block = None
    last_day = max(schedule_dict.keys())
    first_day = min(schedule_dict.keys())

    # Step 1: calculate consecutive workdays from the last block backward to the first off-duty block
    for day in reversed(sorted(schedule_dict.keys())):
        if schedule_dict[day]['shift'] != 'F':  # If it's a workday block (A, N, P)
            last_work_block = schedule_dict[day]
            consecutive_count += last_work_block['length']
            last_day = day
        else:
            break

    # Step 2: Calculate consecutive workdays from the first block up to the first off-duty block
    for day in sorted(schedule_dict.keys()):
        if schedule_dict[day]['shift'] != 'F':  # If it's a workday block (A, N, P)
            first_work_block = schedule_dict[day]
            consecutive_count += first_work_block['length']
            first_day = day
        else:
            break

    # Step 3: If combined consecutive workdays exceed max limit, insert off-duty block
    if consecutive_count > max_consecutive_days and off_block_to_use:
        off_day, off_block = off_block_to_use  # Retrieve the saved off-duty block

        # Insert the off-duty block after the last work block
        adjusted_schedule[last_day + 1] = off_block

        # Remove the original off-duty block location
        del adjusted_schedule[off_day]

    return adjusted_schedule


def insert_off_duty_blocks(work_blocks):
    """
    Inserts off-duty blocks ('F') into a schedule of workday blocks to ensure compliance with roster requirements.
    This function distributes exactly 6 off-duty days across the schedule, with each off-duty block lasting
    between 1 and 3 days. Additionally, at least one off-duty day is scheduled on a weekend (days 5, 6, 12, 13,
    19, or 20), and off-duty blocks are inserted without overwriting existing workday shifts.

    This function also handles circular scheduling and enforces constraints on consecutive
    workdays by breaking up extended sequences where necessary to meet the maximum consecutive workday limit. It
    performs validation and makes adjustments as needed to maintain compliance with all scheduling requirements.

    Args:
    - work_blocks (list of lists): The initial schedule, where each sublist represents a block of consecutive
      workdays in shifts ('A', 'P', 'N').

    Returns:
    - list of lists: The final schedule with off-duty blocks inserted to satisfy all constraints, including circular
      workdays and maximum consecutive workday limits.
    """
    off_duty_blocks = []
    total_off_duty_days = 6
    remaining_off_days = total_off_duty_days
    final_schedule = []
    first_off_day = False

    # Step 1: Decide if the first block should be off-duty
    if random.choice([True, False]) and remaining_off_days > 0 and not remaining_work_day:
        block_size = random.randint(1, min(3, remaining_off_days))  # Random block size between 1 and 3
        final_schedule.append([SHIFT_OFFDUTY] * block_size)  # Insert off-duty block at the start
        remaining_off_days -= block_size
        first_off_day = True

    consecutive_days = 0  # Track consecutive workdays
    ongoing = False  # Track if we are preventing consecutive workday violations

    # Step 2: Insert off-duty blocks between workday blocks to distribute rest days.
    for i in range(len(work_blocks)):

        # Add the current work block to the final schedule
        final_schedule.append(work_blocks[i])

        # Only insert off-duty blocks if off-duty days are still remaining
        if remaining_off_days > 0 and i < len(work_blocks):
            if len(work_blocks[i]) < 3: # Maintain constraints on consecutive workdays.
                consecutive_days = 2
                continue

            # Randomly determine off-duty block size between 0 - 3 or remaining off days.
            block_size = random.randint(0, min(3, remaining_off_days))
            if not ongoing:
                consecutive_days += len(work_blocks[i])

            # Ensure no consecutive workday block exceeds 7 days
            if block_size == 0:
                if i == len(work_blocks) - 1:
                    if not remaining_work_day:
                        consecutive_days += len(work_blocks[0])
                        ongoing = True
                        if consecutive_days > 7:
                            # Prioritize splitting large blocks that exceed 7 days
                            block_size = random.randint(1, min(3, remaining_off_days))

                # Adjust between blocks if consecutive limit is hit.
                elif i < len(work_blocks) - 1:
                    consecutive_days += len(work_blocks[i + 1])
                    ongoing = True
                    if consecutive_days > 7:
                        # If we hit 7 consecutive days, force off-duty insertion
                        block_size = random.randint(1, min(3, remaining_off_days))

            else:
                # Reset consecutive days if an off-duty block is added.
                consecutive_days = 0
                ongoing = False

            # Handle special cases for end-of-schedule off-duty insertion
            if di_remaining_work_day and i == len(work_blocks) - 1:
                block_size = 0
                if len(final_schedule[-2]) < 3:
                    final_schedule[-2].insert(len(final_schedule[-2]), SHIFT_OFFDUTY)
                    remaining_off_days -= 1

            if remaining_work_day and i == len(work_blocks) - 1:
                block_size = 0

            final_schedule.append([SHIFT_OFFDUTY] * block_size)
            remaining_off_days -= block_size

    # Step 3: Distribute any remaining off-duty days, ensuring no block exceeds 3 days.
    if remaining_off_days > 0:
        starting_index = 1
        if remaining_work_day or first_off_day:
            starting_index = 2

        # Place remaining off-duty days in existing off-duty blocks.
        for idx in range(len(final_schedule) - starting_index, -1, -1):
            if all(shift == SHIFT_OFFDUTY for shift in final_schedule[idx]) or len(final_schedule[idx]) == 0:
                available_space = 3 - len(final_schedule[idx])  # Ensure no block exceeds 3 days
                if available_space > 0:
                    to_add = min(remaining_off_days, available_space)
                    final_schedule[idx].extend([SHIFT_OFFDUTY] * to_add)
                    remaining_off_days -= to_add
                    if remaining_off_days == 0:
                        break  # Stop once all off-duty days are inserted

    # Convert the schedule to a dictionary to handle consecutive workdays check
    schedule_dict = convert_schedule_to_dict(final_schedule)

    # Step 4: Validate consecutive workdays, adjusting as necessary to meet constraints.
    if check_consecutive_workdays(final_schedule)[0]:
        schedule_dict = check_and_adjust_consecutive_workdays(schedule_dict)
        final_schedule = convert_dict_to_schedule(schedule_dict)
        # Recheck for circular workday issues if needed.
        if check_consecutive_workdays(final_schedule)[0]:
            schedule_dict = handle_circular_consecutive_workdays(schedule_dict)
            final_schedule = convert_dict_to_schedule(schedule_dict)

    # Step 5: Ensure at least one off-duty day falls on a weekend
    off_duty_weekend = has_off_duty_on_weekend(final_schedule)
    if not off_duty_weekend:
        # If no off-duty day on the weekend, adjust the schedule
        result = find_replaceable_weekend_index(final_schedule)
        if result is not None:
            block_idx, weekend_idx, block_weekend_idx = result
            final_schedule = replace_with_off_duty(final_schedule, block_idx, block_weekend_idx, weekend_idx)

    return final_schedule


def convert_schedule_to_dict(schedule):
    """
    Converts a list-based schedule to a dictionary where each index corresponds to the shift type and length of the block.

    Args:
    - schedule (list of lists): The final schedule where each sub-list is a block of shifts.

    Returns:
    - schedule_dict (dict): A dictionary where the keys are day indices and the values are dictionaries
                            containing 'shift' and 'length' information for each block.
    """
    schedule_dict = {}
    current_day = 0  # Track the day index
    for block in schedule:
        block_length = len(block)
        shift_type = block[0] if block else None  # Get the shift type if the block is not empty

        # Add the block to the dictionary
        if shift_type:
            schedule_dict[current_day] = {"shift": shift_type, "length": block_length}
            current_day += block_length  # Move the current day index forward by the block length

    return schedule_dict


def convert_dict_to_schedule(schedule_dict):
    """
    Converts a list-based schedule to a dictionary format where each day index is mapped to
    a shift type and the length of its corresponding block. This format helps streamline
    validation and adjustment of scheduling constraints.

    Args:
    - schedule (list of lists): The initial schedule, where each sub-list represents a block of consecutive shifts.

    Returns:
    - dict: A dictionary where keys are day indices, and values are dictionaries containing
            'shift' and 'length' information for each block, facilitating constraint checking.
    """
    schedule = []
    current_day = 0  # Track the current day in the final schedule format

    # Sort the schedule_dict by the start day to reconstruct the schedule in order
    sorted_schedule = sorted(schedule_dict.items())

    for day, block in sorted_schedule:
        shift_type = block['shift']
        block_length = block['length']
        # Append the shift type to the schedule as a block of that shift
        schedule.append([shift_type] * block_length)
        current_day += block_length

    return schedule


def consecutive_numbers(list: List, length: int, elements: Set) -> Dict[int, int]:
    """
        Identifies consecutive occurrences of specified elements in a list, supporting circular
        handling by wrapping around the list. This function returns the starting index and
        length of each consecutive block containing only the specified elements.

        Args:
        - list (List): The input list to analyze.
        - length (int): The length of relevant items in the list, accounting for circular behavior.
        - elements (Set): The set of elements to count consecutively.

        Returns:
        - Dict[int, int]: A dictionary where each key is the starting index of a consecutive sequence,
                          and each value is the length of that sequence.
    """
    result = {}

    # We iterate from position 0 to position len(list)*2 and count the number of occurrence.  We reset when we see something else
    have_seen_something_else = False
    nb_consecutives = 0  # Only start counting after we have seen something from outside [elements]

    for i in range(length * 2):
        element = list[i % length]
        if element in elements:
            if have_seen_something_else:
                nb_consecutives += 1

        else:  # element is not in elements
            have_seen_something_else = True
            if nb_consecutives != 0:
                start = i - nb_consecutives
                result[start] = nb_consecutives
            nb_consecutives = 0

    return result


def swap_lists(list_of_lists, index1, index2):
    """
    Swaps two lists within a list of lists at specified indices, returning the updated list.

    Args:
    - list_of_lists (List[List]): The main list containing sublists to be swapped.
    - index1 (int): The index of the first list to swap.
    - index2 (int): The index of the second list to swap.

    Returns:
    - List[List]: The list of lists after swapping the specified sublists.

    Raises:
    - IndexError: If either index1 or index2 is out of the range of list_of_lists.
    """
    # Validate indices to ensure they are within the bounds of list_of_lists
    if index1 >= len(list_of_lists) or index2 >= len(list_of_lists):
        raise IndexError("Index out of range")

    # Perform the swap of the two specified sublists
    list_of_lists[index1], list_of_lists[index2] = list_of_lists[index2], list_of_lists[index1]

    return list_of_lists
