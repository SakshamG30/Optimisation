import pulp
from pulp import LpVariable as Var
import uuid


def pulp_abs(m, x, y, x_lb=None, x_ub=None):
    """
    Implements y = |x|

    x and y can be variables or expressions. Boudns are required on x, which
    can be obtained from x if it is a variable, but otherwise will need to be
    supplied through x_lb and x_ub.

    The required constraints are automatically added to the model, but are also
    returned in case it is desirable to obtain dual information.
    """
    x_lb = x.lowBound if x_lb is None else x_lb
    x_ub = x.upBound if x_ub is None else x_ub

    if x_lb is None or x_ub is None:
        raise ValueError('x is missing a lower or upper no bound')

    u = Var(f'pulp_abs_{uuid.uuid4()}', cat='Binary')
    c1 = y - x >= 0
    c2 = y - x <= -2 * x_lb * u
    c3 = y + x >= 0
    c4 = y + x <= 2 * x_ub * (1 - u)

    m += c1
    m += c2
    m += c3
    m += c4

    return (c1, c2, c3, c4)


def pulp_abs_tests():
    m = pulp.LpProblem()

    x = Var('x', lowBound=-4, upBound=5)
    y = Var('y')

    pulp_abs(m, x, y)
    m += y

    m.solve()
    assert m.status == 1
    assert x.value() == 0
    assert y.value() == 0

    m.objective = -y  # explicitly set objective to override the warning
    m.solve()
    assert m.status == 1
    assert x.value() == 5
    assert y.value() == 5

    m += x <= 3
    m.solve()
    assert m.status == 1
    assert x.value() == -4
    assert y.value() == 4

    m += x <= -2
    m.objective = y  # explicitly set objective to override the warning
    m.solve()
    assert m.status == 1
    assert x.value() == -2
    assert y.value() == 2


if __name__ == '__main__':
    pulp_abs_tests()
    print('Tests complete')
