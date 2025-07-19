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

    # Flow variables for each edge
    flows = [Var(f'flow_{i}_{j}') for i, j in edges]

    # Upgradee variables for each edge
    upgrades = [Var(f'upgrade_{i}_{j}', 0) for i, j in edges]

    # Supply variables for some nodes
    supply = {n: Var(f'sup_{n}', 0) for n in supply_nodes}

    # Get nodes
    nodes = set()
    for i, j in edges:
        nodes.add(i)
        nodes.add(j)

    # Get node to edge mapping
    edges_in = {n: [] for n in nodes}
    edges_out = {n: [] for n in nodes}
    for k, (i, j) in enumerate(edges):
        edges_in[j].append(k)
        edges_out[i].append(k)

    # Impose conservation constraint everywhere
    for n in nodes:
        m += (lpSum(flows[i] for i in edges_in[n])
              - lpSum(flows[i] for i in edges_out[n])
              - demand.get(n, 0.0)
              + supply.get(n, 0.0) == 0)

    # Constrain flows
    for flow, cap, upgrade in zip(flows, capacities, upgrades):
        m += flow <= cap + upgrade
        m += -cap - upgrade <= flow

    # Minimise upgrade costs
    m += lpSum(upgrades)

    m.solve()
    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Objective: {m.objective.value()}')
    print('Upgrades:')
    print([upgrade.value() for upgrade in upgrades])


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
