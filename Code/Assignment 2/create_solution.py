# Copyright subsists in all the material on this repository and vests in the ANU
# (or other parties). Any copying of the material from this site other than for
# the purpose of working on this assignment is a breach of copyright.
#
# In practice, this means that you are not allowed to make any of the existing
# material in this repository nor your assignment available to others (except to
# the course lecturers and tutors) at any time before, during, or after the
# course. In particular, you must keep any clone of this repository and any
# extension of it private at all times.
import time
from sys import argv
import random
from nurse import RosteringProblem, save_roster
from solution2 import validate_and_save


def create_solution(seed):
    '''
    Creates a feasible nurse schedule based on the specified seed. Different seeds lead to
    varied solutions, and the function retries generating a valid schedule if necessary.

    Args:
    - seed (int): Random seed value to ensure reproducibility of generated schedules.

    Returns:
    - nurses (List[str]): A list of validated nurse schedules that satisfy all constraints.
    - m_nurses (int): The count of valid nurse schedules found.
    '''
    random.seed(seed)

    num_combinations = 1101 # Initial number of combinations to validate in each batch
    m_nurses = 0  # To store the count of valid schedules found
    nurses = None
    count = 0 # Retry counter

    # Try up to 10 times to generate a valid solution before increasing the number of combinations
    while m_nurses == 0 and count < 10:
        nurses, m_nurses = validate_and_save(num_combinations=400000, check_num_combinations=num_combinations)
        count += 1
        num_combinations += 10

    return nurses, m_nurses


if __name__ == '__main__':
    arg = 0  # default seed
    if len(argv) > 1:
        arg = int(argv[1])

    roster, min_nb_nurses = create_solution(arg)

    # Sanity check: making sure the roster is feasible
    prob = RosteringProblem()
    feasibility = prob.is_feasible(roster)
    if feasibility is not None:
        print(f'Roster is not feasible ({feasibility})')
        exit(0)

    save_roster(roster)
