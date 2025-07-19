import pulp
from pulp import LpVariable as Var


def expansion(edges, capacities, supply_nodes, demand, optional):
    """
    Directed graph, but where flow can be in either direction.
    Nodes given implictly.
    Demand is fixed demand at nodes.
    Optional gives the unbuilt edges and cost of building
    """
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Flow variables for each edge
    flows = [Var('flow_{}_{}'.format(i, j)) for i, j in edges]

    # Expansion variables for optional edges
    expand = {opt: Var('expand_{}'.format(opt), cat='Binary')
              for opt in optional}

    # Supply variables for some nodes
    supply = {n: Var('sup_{}'.format(n), lowBound=0) for n in supply_nodes}

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
        m += (sum(flows[i] for i in edges_in[n])
              - sum(flows[i] for i in edges_out[n])
              - demand.get(n, 0.0)
              + supply.get(n, 0.0) == 0)

    # Constrain flows
    for k, (flow, cap) in enumerate(zip(flows, capacities)):
        # Only allow flow if expanded
        if k in optional:
            m += flow <= cap * expand[k]
            m += -cap * expand[k] <= flow
        else:
            m += flow <= cap
            m += -cap <= flow

    # Minimise expansion costs
    m += sum(expand[k] * optional[k] for k in optional)

    # print(m)
    m.solve(solver=pulp.COIN_CMD(msg=1))
    return {
        'status': m.status,
        'objective': m.objective.value(),
        'expanded': {edges[k]: expand[k].value() for k in expand},
    }


res = expansion(
    edges=[
        ('moisture', 'hotter'), ('hotter', 'thirsty'), ('thirsty', 'parched'),
        ('parched', 'arid'), ('moisture', 'crispy'), ('crispy', 'thirsty'),
        ('crispy', 'torched'), ('torched', 'arid'), ('torched', 'dusty'),
        ('dusty', 'arid'), ('crispy', 'crackle'), ('crackle', 'torched'),
        ('crackle', 'baked'), ('baked', 'dusty'), ('moisture', 'heatstroke'),
        ('heatstroke', 'crackle'),
        ('restless', 'hotter'), ('restless', 'thirsty'),
        ('surplus', 'arid'),
        ('desal', 'crackle'), ('desal', 'torched'),
        ('hotter', 'crispy'), ('restless', 'parched'), ('surplus', 'parched'),
        ('desal', 'dusty'), ('thirsty', 'torched')
    ],
    capacities=[1, 2, 3, 14, 16, 3, 5, 6, 15, 12, 10, 8, 4, 3, 10, 5,
                3, 6, 5, 8, 8, 10, 10, 10, 10, 10],
    supply_nodes=['moisture', 'restless', 'surplus', 'desal'],
    demand={
        'hotter': 6,
        'thirsty': 7,
        'parched': 7,
        'arid': 12,
        'crispy': 12,
        'torched': 4,
        'dusty': 4,
        'crackle': 8,
        'baked': 5,
        'heatstroke': 6,
    },
    optional={
        21: 2,
        22: 3,
        23: 3,
        24: 1,
        25: 4,
    },
)
print(res)
