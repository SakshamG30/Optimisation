import pulp
from pulp import LpVariable as Var


def brewery(styles, ingredients, fermenter_capacity, fixed_costs):
    """
    """
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Variables
    beer = {s: Var(f'b_{s}', lowBound=styles[s]['min'])
            for s in styles}
    ingr = {i: Var(f'ingr_{i}', lowBound=0, upBound=ingredients[i]['max'])
            for i in ingredients}

    # Constraints
    for i in ingredients:
        m += sum(styles[s]['recipe'].get(i, 0.0) * beer[s]
                 for s in styles) == ingr[i]

    m += sum(beer.values()) <= fermenter_capacity

    # Objective
    m += (fixed_costs
          + sum(ingredients[i]['price'] * ingr[i] for i in ingredients)
          - sum(styles[s]['price'] * beer[s] for s in styles))

    # print(m)
    # m.solve(solver=pulp.COIN_CMD(msg=1))
    m.solve(solver=pulp.PULP_CBC_CMD(msg=0))
    return (m.status,
            m.objective.value(),
            {s: b.value() for s, b in beer.items()},
            {i: g.value() for i, g in ingr.items()},
            )


res = brewery(
    styles={
        'ipa': {
            'price': 6.4,
            'recipe': {
                'water': 1.5,
                'wheat': 0.025,
                'barley': 0.25,
                'casc': 0.0030,
                'tett': 0.0005,
            },
            'min': 0.0,
        },
        'hef': {
            'price': 6.1,
            'recipe': {
                'water': 1.3,
                'wheat': 0.2,
                'barley': 0.025,
                'tett': 0.001,
            },
            'min': 0.0,
        },
        'sto': {
            'price': 6.2,
            'recipe': {
                'water': 1.8,
                'wheat': 0.025,
                'barley': 0.35,
                'tett': 0.002,
            },
            'min': 1000.0, # Question 1 set to 500
                           # Question 2 set to 0
        },
    },
    ingredients={
        'water': {'price': 0.01, 'max': 5000},  # L
        'barley': {'price': 0.8, 'max': 500},  # kg
        'wheat': {'price': 0.7, 'max': 400},  # kg
        'casc': {'price': 70, 'max': 7},  # kg
        'tett': {'price': 60, 'max': 4},  # kg
    },
    fermenter_capacity=3500,  # L
    fixed_costs=17000,  # $ Question 2 add 1000
)

print('Status: {} Objective: {}'.format(res[0], res[1]))
print(res[2])
print(res[3])
