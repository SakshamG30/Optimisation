import pulp
from pulp import LpVariable as Var
from pulp import lpSum


def min_upgrade(edges, capacities, supply_nodes, demand):
    """
    Directed graph, but where flow can be in either direction.
    Nodes given implictly.
    Demand is fixed demand at nodes.
    """
    m = pulp.LpProblem()

    # Modelling here.
    # You can use the modelling in water_supply.py as a starting point, and
    # modify it to suit the upgrade problem.
    # See the end of LP 3 Optimality lecture slides for further details.

    upgrades = None  # should be a list of upgrade variables
    
    m += 0

    m.solve()
    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Objective: {m.objective.value()}')
    print('Upgrades:')


# Expect zero upgrades for this (demand is feasible)
instance_low_demand = {
    "edges": [
        ('moisture', 'hotter'), ('hotter', 'thirsty'), ('thirsty', 'parched'),
        ('parched', 'arid'), ('moisture', 'crispy'), ('crispy', 'thirsty'),
        ('crispy', 'torched'), ('torched', 'arid'), ('torched', 'dusty'),
        ('dusty', 'arid'), ('crispy', 'crackle'), ('crackle', 'torched'),
        ('crackle', 'baked'), ('baked', 'dusty'), ('moisture', 'heatstroke'),
        ('heatstroke', 'crackle'),
        ('restless', 'hotter'), ('restless', 'thirsty'),
        ('surplus', 'arid'),
        ('desal', 'crackle'), ('desal', 'torched'),
    ],
    "capacities": [1, 2, 3, 14, 16, 3, 5, 6, 15, 12, 10, 8, 4, 3, 10, 5, 3, 6,
                   5, 8, 8],
    "supply_nodes": ['moisture', 'restless', 'surplus', 'desal'],
    "demand": {
        'hotter': 3,
        'thirsty': 5,
        'parched': 5,
        'arid': 10,
        'crispy': 8,
        'torched': 2,
        'dusty': 2,
        'crackle': 6,
        'baked': 3,
        'heatstroke': 4,
    },
}

instance_high_demand = {
    "edges": [
        ('moisture', 'hotter'), ('hotter', 'thirsty'), ('thirsty', 'parched'),
        ('parched', 'arid'), ('moisture', 'crispy'), ('crispy', 'thirsty'),
        ('crispy', 'torched'), ('torched', 'arid'), ('torched', 'dusty'),
        ('dusty', 'arid'), ('crispy', 'crackle'), ('crackle', 'torched'),
        ('crackle', 'baked'), ('baked', 'dusty'), ('moisture', 'heatstroke'),
        ('heatstroke', 'crackle'),
        ('restless', 'hotter'), ('restless', 'thirsty'),
        ('surplus', 'arid'),
        ('desal', 'crackle'), ('desal', 'torched'),
    ],
    "capacities": [1, 2, 3, 14, 16, 3, 5, 6, 15, 12, 10, 8, 4, 3, 10, 5, 3, 6,
                   5, 8, 8],
    "supply_nodes": ['moisture', 'restless', 'surplus', 'desal'],
    "demand": {
        'hotter': 6,
        'thirsty': 8,
        'parched': 8,
        'arid': 15,
        'crispy': 12,
        'torched': 4,
        'dusty': 7,
        'crackle': 9,
        'baked': 6,
        'heatstroke': 7,
    },
}

if __name__ == "__main__":
    min_upgrade(**instance_low_demand)
    min_upgrade(**instance_high_demand)
