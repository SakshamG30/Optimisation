import pulp
from pulp import LpVariable as Var


def part_1():
    # Create model with default solver
    m = pulp.LpProblem()

    # Create unbounded variables
    x = Var('x')
    y = Var('y')
    z = Var('z')

    # Add constraints to model
    # Not required in this question, but keeping reference to constraints to
    # get duals for comparison in problem with dual reformulation.
    c1 = x + 2 * y + 3 * z >= 2
    c2 = y >= x + 2 * z
    c3 = z <= 6
    m += c1
    m += c2
    m += c3

    # Set objective of model
    m += x + 2 * y - 5 * z

    m.solve()

    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Obj: {m.objective.value()}')
    print(f'x, y, z: {(x.value(), y.value(), z.value())}')
    print(f'duals: {(c1.pi, c2.pi, c3.pi)}')


def part_2():
    # Create model with default solver
    m = pulp.LpProblem()

    # Create non-zero variables
    xn = Var('xn', 0)
    xp = Var('xp', 0)
    yn = Var('yn', 0)
    yp = Var('yp', 0)
    zn = Var('zn', 0)
    zp = Var('zp', 0)
    s1 = Var('s1', 0)
    s2 = Var('s2', 0)
    s3 = Var('s3', 0)

    x = xp - xn
    y = yp - yn
    z = zp - zn

    # Add constraints to model
    m += x + 2 * y + 3 * z - s1 == 2
    m += y - s2 == x + 2 * z
    m += z + s3 == 6

    # Set objective of model
    m += x + 2 * y - 5 * z

    m.solve(solver=pulp.COIN_CMD(msg=1))

    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Obj: {m.objective.value()}')
    print(f'x, y, z: {(x.value(), y.value(), z.value())}')


if __name__ == "__main__":
    part_1()
    part_2()
