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

from nurse import RosteringProblem, load_last_roster, read_costs, save_roster
from neighbours import (ShiftBlockReverseNeighbourhood, SwapNeighbourhood, DoubleSwapNeighbourhood, OneShiftChangeNeighbourhood, \
                        BlockShiftsSwapNeighbourhood, CycleShiftsNeighbourhood,
                        AlternatingShiftAndReverseBlockNeighbourhood, RandomizedBlockSwapAndReverseNeighbourhood)


def local_search(problem, cost_matrix, initial_solution, neighborhood):
    """
    Executes a local search using the provided neighborhood structure until a local optimum is reached.
    The search iteratively explores neighboring solutions and updates to a neighboring solution if it
    offers a lower cost, continuing until no further improvement can be found.

    Args:
        problem (RosteringProblem): The rostering problem instance with constraints and utility functions.
        cost_matrix (dict): A dictionary mapping cost values to shifts within the roster.
        initial_solution (List[str]): The initial feasible roster solution as a list of nurse schedules.
        neighborhood (Neighbourhood): A neighborhood object defining the way neighbors of the current solution are generated.

    Returns:
        Tuple[List[str], float]: The optimized solution roster and the associated cost after reaching a local optimum.
    """
    current_solution = initial_solution
    current_cost = problem.cost(current_solution, cost_matrix)
    improvement_found = True
    iteration_count = 0  # For tracking the number of iterations during local search

    # Continue searching as long as improvements are found
    while improvement_found:
        improvement_found = False
        for neighbor in neighborhood.neighbours(current_solution):
            # Validate the feasibility of the generated neighbor
            if problem.is_feasible(neighbor) is None:
                neighbor_cost = problem.cost(neighbor, cost_matrix)
                # Update if the neighbor offers a cost reduction
                if neighbor_cost < current_cost:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    improvement_found = True
                    break  # Exit loop after finding improvement
        iteration_count += 1

    return current_solution, current_cost


def shake(problem, neighborhood, current_solution, max_random_neighbors=4):
    """
    Applies a shake operation to the current solution, introducing randomness by exploring a limited
    number of neighboring solutions and selecting the one with the lowest cost among feasible neighbors.

    Args:
        problem (RosteringProblem): An instance of the rostering problem, containing the constraints and utility functions.
        neighborhood (Neighbourhood): The neighborhood object that generates neighbors for the solution.
        current_solution (List[str]): The current feasible roster solution as a list of nurse schedules.
        max_random_neighbors (int): The maximum number of randomly selected neighbors to evaluate. Default is 4.

    Returns:
        List[str]: A modified solution selected from feasible neighbors, or the original solution if no improvement is found.
    """
    best_neighbor = None
    lowest_cost = float('inf')
    improvement_found = False  # Flag to track if an improvement is found during shaking

    # Generate and shuffle neighbors to introduce randomness
    potential_neighbors = neighborhood.neighbours(current_solution)
    random.shuffle(potential_neighbors)  # Shuffle to randomize order

    # Limit neighbors to the maximum specified number
    selected_neighbors = potential_neighbors[:max_random_neighbors]

    for neighbor in selected_neighbors:
        # Check feasibility of each neighbor
        if problem.is_feasible(neighbor) is None:
            neighbor_cost = problem.cost(neighbor, costs)
            # Update if this neighbor has a lower cost
            if neighbor_cost < lowest_cost:
                best_neighbor = neighbor
                lowest_cost = neighbor_cost
                improvement_found = True  # Improvement found

    return best_neighbor if best_neighbor else current_solution


def variable_neighbourhood_search(problem, cost_matrix, initial_roster, neighborhoods, max_vns_iterations=1000,
                                  max_no_progress=220):
    """
    Perform Variable Neighbourhood Search (VNS) to optimize a roster solution by iteratively applying
    shaking and local search techniques across multiple neighbourhoods until no further improvement is found.

    Args:
        problem (RosteringProblem): The rostering problem instance containing constraints and parameters.
        cost_matrix (dict): A dictionary mapping roster configurations to associated costs.
        initial_roster (List[str]): The initial feasible roster of nurse schedules.
        neighborhoods (List[Neighbourhood]): List of neighbourhood structures to explore.
        max_vns_iterations (int): Maximum number of VNS iterations. Default is 1000.
        max_no_progress (int): Maximum allowed iterations without improvement. Default is 200.

    Returns:
        Tuple (List[str], float): Returns the optimal roster and its associated cost.
    """
    # Initialize with the starting solution and cost
    active_solution = initial_roster
    active_cost = problem.cost(active_solution, cost_matrix)
    optimal_solution = active_solution
    optimal_cost = active_cost

    no_progress_iterations = 0  # Tracks the number of consecutive non-improving iterations

    # Begin the VNS loop, iterating through up to max_vns_iterations
    for vns_round in range(max_vns_iterations):
        neighborhood_index = 0  # Start at the first neighbourhood

        # Cycle through all neighbourhoods
        while neighborhood_index < len(neighborhoods):

            # Shaking phase: Apply random changes within the current neighbourhood
            candidate_solution = shake(problem, neighborhoods[neighborhood_index], active_solution, len(neighborhoods) - 2)

            # Check if the candidate solution from shake is feasible before local search
            if problem.is_feasible(candidate_solution) is not None:
                neighborhood_index += 1
                continue  # Move to the next neighborhood if infeasible

            # Local search phase to refine the solution within the current neighborhood
            local_optimum, local_optimum_cost = local_search(problem, cost_matrix, candidate_solution, neighborhoods[neighborhood_index])

            # Go to the next neighborhood after local search
            neighborhood_index += 1

            # Check if local search improved the current solution
            if local_optimum_cost < active_cost:
                active_solution = local_optimum
                active_cost = local_optimum_cost
                neighborhood_index = 0  # Reset to the first neighborhood after a better solution than the local optimum

                # If this is the best solution found so far, update optimal solution and cost
                if active_cost < optimal_cost:
                    optimal_solution = active_solution
                    optimal_cost = active_cost
                    save_roster(optimal_solution)
                    no_progress_iterations = 0  # Reset no improvement counter
                else:
                    no_progress_iterations += 1
            else:
                no_progress_iterations += 1

        # Terminate if no improvement for max_no_progress iterations
        if no_progress_iterations >= max_no_progress:
            break

    return optimal_solution, optimal_cost



if __name__ == '__main__':

    prob = RosteringProblem()
    costs = read_costs(prob)
    initial_solution = load_last_roster()
    initial_solution = [line.strip() for line in initial_solution]

    neighbourhoods = [
        RandomizedBlockSwapAndReverseNeighbourhood(prob),
        AlternatingShiftAndReverseBlockNeighbourhood(prob),
        ShiftBlockReverseNeighbourhood(prob),
        SwapNeighbourhood(prob),
        CycleShiftsNeighbourhood(prob),
        BlockShiftsSwapNeighbourhood(prob),
        OneShiftChangeNeighbourhood(prob),
        DoubleSwapNeighbourhood(prob)
    ]

    best_solution, best_cost = variable_neighbourhood_search(prob, costs, initial_solution, neighbourhoods)

    print(f"VNS completed. Best solution cost: {best_cost}")

    # Check final solution feasibility before saving
    if prob.is_feasible(best_solution) is None:
        save_roster(best_solution)
