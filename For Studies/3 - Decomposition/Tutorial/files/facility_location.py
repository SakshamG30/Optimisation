#!/usr/bin/env python3

import pulp
from pulp import LpVariable as Var

class FacilityLocationBendersRMP:
    def __init__(self, facility_costs):
        self.rmp = pulp.LpProblem(sense=pulp.LpMinimize)
        self.n_facilities = len(facility_costs)
        self.facility_costs = facility_costs
        # Inicates whether or not each facility is built (y_k in the slides)
        self.facility = [Var(f'y_{k}', cat='Binary') for k in range(self.n_facilities)]
        self.eta = Var('eta')

        # TO BE IMPLEMENTED
        # Add you extra constraint here that guarantees feasibility of the subproblems
        raise NotImplementedError

        # Objective
        self.rmp += pulp.lpSum(self.facility[k] * facility_costs[k] for k in range(self.n_facilities)) + self.eta


    def solve(self):
        # print(self.rmp)
        self.rmp.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        incumbent_rmp_solution = [self.facility[i].value() for i in range(self.n_facilities)]
        # print(f'incumbent: {incumbent_rmp_solution}')
        # print(f'eta = {self.eta.value()}')
        lower_bound = self.rmp.objective.value()
        return incumbent_rmp_solution, lower_bound


    def add_optimality_cut(self, values_for_cut):
        # TO BE IMPLEMENTED
        raise NotImplementedError


    def solution_cost(self, incumbent_rmp_solution, values_for_cut):
        """ Returns the cost of the solution given by the current incumbent and the value of the cut"""
        # TO BE IMPLEMENTED
        raise NotImplementedError


class FacilityLocationBendersSubProblem:
    def __init__(self, transport_costs):
        self.subproblem = pulp.LpProblem(sense=pulp.LpMinimize)
        self.n_customers = len(transport_costs)
        self.n_facilities = len(transport_costs[0])

        # proportion of demand of customer i served by facility at k
        self.customer = [[Var(f'x_{i}_{k}'.format(i,k), lowBound=0) for k in range(self.n_facilities)] for i in range(self.n_customers)]

        # TO BE IMPLEMENTED
        # Build the **primal** subproblem using the variables in self.customer in the self.subproblem model
        raise NotImplementedError


    def solve(self, incumbent_rmp_solution):
        """ Update the primal subproblem based on the incumbent_rmp_solution (see pdf for hints)
            and returns the status of the optimization and the values for the cut"""
        # TO BE IMPLEMENTED
        raise NotImplementedError



def benders_decomposition(facility_costs, transport_costs, tolerance=1e-6):
    upper_bound = float("Inf")
    lower_bound = float("-Inf")

    n_facilities = len(facility_costs)
    # Starting by building just the first facility
    incumbent_rmp_solution = [0] * n_facilities
    incumbent_rmp_solution[0] = 1

    benders_rmp = FacilityLocationBendersRMP(facility_costs)
    benders_subproblem = FacilityLocationBendersSubProblem(transport_costs)

    iteration = 0
    while upper_bound - lower_bound > tolerance:
        # print(f'[iteration {iteration}] LB = {lower_bound} UB = {upper_bound}')
        # print(f'incumbent: {incumbent_rmp_solution}')
        solution_type, values_for_cut = benders_subproblem.solve(incumbent_rmp_solution)

        # We simplified the problem so that only optimality cuts are needed.
        # If the solution_type is not LpStatusOptimal, then there is something wrong in the model
        assert solution_type == pulp.LpStatusOptimal

        # TO BE IMPLEMENTED
        raise NotImplementedError
        iteration += 1

    print(f'[DONE] LB = {lower_bound} UB = {upper_bound}')
    return upper_bound, incumbent_rmp_solution



def single_mip_solution(facility_costs, transport_costs):
    """ This method solves the problem directly with a single MIP. Use this as a reference solver
        to double check you got the right solution"""

    m = pulp.LpProblem(sense=pulp.LpMinimize)

    n_facilities = len(facility_costs)
    n_customers = len(transport_costs)

    # Inicates whether or not each facility is built (y_k in the slides)
    facility = [Var(f'y_{k}', cat='Binary') for k in range(n_facilities)]

    # proportion of demand of customer i served by facility at k
    customer = [[Var(f'x_{i}_{k}'.format(i,k), lowBound=0, upBound=1) for k in range(n_facilities)] for i in range(n_customers)]

    # All customers demand must be met
    for i in range(n_customers):
        m += pulp.lpSum(customer[i]) == 1

    # Can only attend customer if facility is built built
    for k in range(n_facilities):
        # m += pulp.lpSum(customer[i][k]) <= facility[k]
        for i in range(n_customers):
            m += customer[i][k] <= facility[k]

    m += (pulp.lpSum(facility[k] * facility_costs[k] for k in range(n_facilities))
          + pulp.lpSum(
                pulp.lpSum(customer[i][k] * transport_costs[i][k] for k in range(n_facilities))
            for i in range(n_customers))
          )

    # print(m)
    m.solve(solver=pulp.PULP_CBC_CMD(msg=0))
    facilities_used = [i for i in range(n_facilities) if facility[i].value() == 1.0]
    print(f'status = {m.status}, obj = {m.objective.value()}, facilities built = {facilities_used}')
    return m.objective.value(), [facility[i].value() for i in range(n_facilities)]


def main():
    # Costs of building a facility at location k (b_k in the slides)
    facility_costs = [162, 261, 103]

    # Costs of transport from customer i to facility k (c_ik in the slides)
    transport_costs = [
        [20, 20, 30],
        [10, 10, 40],
        [25,  5, 25],
        [15, 20, 10],
         [5, 25, 10],
    ]
    for client_cost in transport_costs:
        assert len(client_cost) == len(facility_costs)

    b_obj, b_facilities = benders_decomposition(facility_costs, transport_costs)
    mip_obj, mip_facilities = single_mip_solution(facility_costs, transport_costs)

    assert b_obj == mip_obj

if __name__ == "__main__":
    main()
