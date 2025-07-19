import pulp
from pulp import LpVariable as Var
from pulp import lpSum


def min_cost_supply(edges, capacities, prices, demand):
    """
    Directed graph, but where flow can be in either direction.
    Nodes given implictly.
    Price of production at nodes.
    Demand is fixed demand at nodes.
    """
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    # Flow variables for each edge
    flows = [Var(f'flow_{i}_{j}', -cap, cap)
             for (i, j), cap in zip(edges, capacities)]

    # Supply variables for some nodes
    supply = {n: Var(f'sup_{n}', 0) for n in prices}

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

    # Minimise supply costs
    m += lpSum(s * prices[n] for n, s in supply.items())

    # print(m)
    # m.solve(solver=pulp.COIN_CMD(msg=1))
    m.solve()
    return (m.status, [flow.value() for flow in flows], m.objective.value())


res = min_cost_supply(
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
    ],
    capacities=[1, 2, 3, 14, 16, 3, 5, 6, 15, 12, 10, 8, 4, 3, 10, 5,
                3, 6, 5, 8, 8],
    prices={'moisture': 1, 'restless': 1, 'surplus': 2, 'desal': 4},
    demand={
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
)
print(res)


# The following should be infeasible as it is extra demand for expansion case
#res = min_cost_supply(
#    edges=[
#        ('moisture', 'hotter'), ('hotter', 'thirsty'), ('thirsty', 'parched'),
#        ('parched', 'arid'), ('moisture', 'crispy'), ('crispy', 'thirsty'),
#        ('crispy', 'torched'), ('torched', 'arid'), ('torched', 'dusty'),
#        ('dusty', 'arid'), ('crispy', 'crackle'), ('crackle', 'torched'),
#        ('crackle', 'baked'), ('baked', 'dusty'), ('moisture', 'heatstroke'),
#        ('heatstroke', 'crackle'),
#        ('restless', 'hotter'), ('restless', 'thirsty'),
#        ('surplus', 'arid'),
#        ('desal', 'crackle'), ('desal', 'torched'),
#    ],
#    capacities=[1, 2, 3, 14, 16, 3, 5, 6, 15, 12, 10, 8, 4, 3, 10, 5,
#                3, 6, 5, 8, 8],
#    prices={'moisture': 1, 'restless': 1, 'surplus': 2, 'desal': 4},
#    demand={
#        'hotter': 6,
#        'thirsty': 8,
#        'parched': 8,
#        'arid': 15,
#        'crispy': 12,
#        'torched': 4,
#        'dusty': 7,
#        'crackle': 9,
#        'baked': 6,
#        'heatstroke': 7,
#    },
#)
#print(res)
