#!/usr/bin/env python3

import pulp


class CuttingStockRMP:
    def __init__(self, item_len, item_demand, initial_pattern):
        self.item_demand = item_demand
        self.num_items = len(item_len)
        self.master = pulp.LpProblem("master")
        self.x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(self.num_items)]
        self.master += pulp.lpSum(self.x[i] for i in range(self.num_items))

        # We will use col to represent all columns (patterns) in the RMP so far
        self.col = initial_pattern

        # constraints
        # COMMON PITFALL: do not save the constraint because we will overwrite them in the LP level, so
        # the list/dict holding the constraints would be outdated
        for i in range(self.num_items):
            constr = (pulp.lpSum([self.col[i][k] * self.x[k] for k in range(len(self.col[0]))]) >= item_demand[i])
            # Naming the constraints to be able to overwrite them
            self.master += constr, f"cons{i}"


    def solve(self):
        """ Solves the RMP and return a list of the dual variables """
        self.master.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        print(f"[RMP] Solved.  Status = {pulp.LpStatus[self.master.status]}, objective = {pulp.value(self.master.objective)}")
        return [self.master.constraints[f"cons{i}"].pi for i in range(self.num_items)]


    def add_column(self, new_column):
        self.col.append(new_column)
        num_cols = len(self.col)

        new_var = pulp.LpVariable(f"x{num_cols}", lowBound=0)
        self.x.append(new_var)

        # Add new var to objective
        self.master += self.master.objective + new_var

        # Also add to constraints
        for i in range(self.num_items):
            # Overwriting the constraints -- Other solvers have more efficient interfaces to
            # do so, e.g., Gurobi and CPlex have a column class specific for doing this
            self.master.constraints[f'cons{i}'] = pulp.lpSum([self.col[k][i] * self.x[k] for k in
                range(num_cols)]) >= self.item_demand[i]


    def print_solution(self):
        for k in range(len(self.x)):
           if self.x[k].varValue > 0:
              print(f" {self.x[k].varValue} of the pattern {k} =  {self.col[k]}")


class CuttingStockPrincingSubproblem:
    def __init__(self, rod_len, item_len):
        self.sub = pulp.LpProblem("sub", pulp.LpMaximize)
        num_items = len(item_len)

        # sub vars are a_i = number of item i in a pattern
        self.a = pulp.LpVariable.dicts("a", (i for i in range(num_items)), cat="Integer", lowBound=0)
        self.sub += pulp.lpSum([item_len[i] * self.a[i] for i in range(num_items)]) <= rod_len

        # No objective function because it will be built in the solve method since it depends on the
        # dual variables

    def solve(self, dual_variables):
        num_items = len(dual_variables)

        # Updating the objective function
        self.sub += pulp.lpSum([dual_variables[i] * self.a[i] for i in range(num_items)])
        self.sub.solve(solver=pulp.PULP_CBC_CMD(msg=0))
        print(f"[sub] Solved.  Status = {pulp.LpStatus[self.sub.status]}, objective = {pulp.value(self.sub.objective)}")
        return self.sub.objective.value(), [self.a[i].varValue for i in range(num_items)]


def create_initial_pattern(rod_len, item_len):
    num_items = len(item_len)
    # initial patterns/columns: all zeros except in the diagonal, i.e., 1 cut of 1 length in each
    # pattern
    col = [[0]*num_items for i in range(num_items)]
    for i in range(num_items):
        col[i][i] = 1
    return col


def solve_cutting_stock_cg(rod_len, item_len, item_demand, initial_pattern):
    rmp = CuttingStockRMP(item_len, item_demand, initial_pattern)
    pricing_sub = CuttingStockPrincingSubproblem(rod_len, item_len)

    it = 0
    while True:
        it = it + 1
        dual_vars = rmp.solve()
        princing_sub_obj, new_pattern = pricing_sub.solve(dual_vars)
        # print("f    New pattern is {new_pattern}")
        if princing_sub_obj > 1:
            # Negative reduced cost
            rmp.add_column(new_pattern)
        else:
            # No column with negative reduced cost -- Found the optimal solution
            print(f"Finished after {it} iterations. Optimal solution is: ")
            rmp.print_solution()
            break


def main():
    # Example from the slides
    rod_len = 218
    item_len = [81, 70, 68]
    item_demand = [44, 3, 48]
    initial_pattern = create_initial_pattern(rod_len, item_len)
    solve_cutting_stock_cg(rod_len, item_len, item_demand, initial_pattern)


if __name__ == '__main__':
    main()
