import pulp
from pulp import LpVariable as Var


def problem():
    # Create model with default solver (pulp.LpMinimize, pulp.LpMaximize)
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Create variables (names must be unique) with optional bounds
    x = Var('x', lowBound=None, upBound=None)

    # Add constraints to model (>=, <=, ==)
    m += x >= 5

    # Set objective of model
    m += 2 * x

    # Ensure variables all have unique names
    assert m.checkDuplicateVars()

    # Optionally print out the model to see if what you have built makes sense
    print(m)

    # Solve with default solver (solver messages should indicate CBC)
    m.solve()

    # Status and results of the solve process
    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Obj: {m.objective.value()}, x: {x.value()}')

    # Debugging instructions (IDEs will have their own process).
    # If you want to debug at a specific point in your code, you can create
    # breakpoints like so:
    # breakpoint()
    # You can then interactively inspect variables in the local and global
    # scope, or call `continue` to continue execution up to the next breakpoint
    # in the code. Inspecting variables in the debugger can be slow, so if
    # needed you can call `interact` to drop into an interactive session at
    # that point without the overhead of the debugger. Ctrl-D should get you
    # back to the debugger from the interactive session, but it depends on your
    # environment.


if __name__ == "__main__":
    problem()
