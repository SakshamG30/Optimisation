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
from nurse import RosteringProblem, DAYS_PER_WEEK, SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT

if __name__ == '__main__':
    SEED = 1818576
    random.seed(SEED)

    with open('costs.rcosts', 'w') as f:
        prob = RosteringProblem()
        for i in range(prob._nb_nurses):
            for d in range(DAYS_PER_WEEK):
                for shift_type in [SHIFT_AFTERNOON, SHIFT_MORNING, SHIFT_NIGHT]:
                    f.write(str(random.random())) # chooses a random number between 0 and 1
                    f.write(' ')
