import pulp
from pulp import LpVariable as Var


def locate(sites, trips, serve):
    """
    Locate drone hangers and allocate capacity to restaurants.
    """
    m = pulp.LpProblem(sense=pulp.LpMinimize)

    nr = len(trips)  # number of restaurants

    # Inicates whether or not each hanger is built
    hanger = {s: Var('h_{}'.format(s), cat='Binary') for s in sites}
    # Amount each hanger delivers for each restaurant
    deliver = {s: [
        Var('d_{}_{}'.format(s, r), lowBound=0)
        for r in range(nr)] for s in sites}

    # Can only deliver if hanger built
    for s in sites:
        m += sum(deliver[s]) <= sites[s]['capacity'] * hanger[s]

    # Demand must be met
    for r in range(nr):
        m += sum(deliver[s][r] for s in sites) == trips[r]

    # Cost to building hangers and serve restaurants
    m += (sum(sites[s]['cost'] * hanger[s] for s in sites)
          + sum(sum(
              serve[s][r] * deliver[s][r]
              for r in range(nr)) for s in sites)
          )

    #print(m)
    m.solve(solver=pulp.COIN_CMD(msg=1))
    sites_used = set([s for s in sites if hanger[s].value() == 1.0])
    delivered = {s: [deliver[s][r].value() for r in range(nr)] for s in sites}
    return {
        'status': m.status,
        'objective': m.objective.value(),
        'sites': sites_used,
        'delivered': {s: [100 * delivered[s][r] / trips[r] for r in range(nr)]
                      for s in sites_used},
    }


res = locate(
    sites={
        'A': {'cost': 42, 'capacity': 32},
        'B': {'cost': 63, 'capacity': 19},
        'C': {'cost': 36, 'capacity': 20},
        'D': {'cost': 75, 'capacity': 35},
        'E': {'cost': 48, 'capacity': 42},
        'F': {'cost': 60, 'capacity': 40},
        'G': {'cost': 72, 'capacity': 18},
        'H': {'cost': 75, 'capacity': 26},
        'I': {'cost': 33, 'capacity': 48},
    },
    trips=[14, 11, 2, 5, 9, 7, 12, 4, 6, 7, 10, 7, 12, 13, 2, 5, 9, 6],
    serve={
        'A': [2, 3, 3, 4, 4, 6, 5, 6, 6, 5, 6, 7, 9, 8, 8, 7, 8, 8],
        'B': [2, 2, 3, 3, 3, 6, 4, 6, 6, 5, 6, 7, 9, 7, 8, 7, 7, 8],
        'C': [2, 3, 4, 4, 4, 5, 5, 5, 5, 4, 5, 6, 8, 7, 7, 6, 6, 7],
        'D': [4, 5, 6, 6, 6, 5, 5, 4, 4, 2, 3, 5, 6, 5, 5, 5, 5, 6],
        'E': [4, 5, 6, 6, 5, 5, 5, 3, 3, 2, 3, 4, 7, 5, 6, 4, 5, 6],
        'F': [4, 6, 7, 7, 6, 5, 6, 4, 4, 1, 2, 5, 5, 5, 5, 4, 4, 6],
        'G': [5, 6, 7, 7, 6, 6, 6, 4, 4, 2, 3, 5, 6, 4, 5, 3, 4, 5],
        'H': [5, 6, 7, 7, 7, 6, 6, 5, 5, 2, 2, 5, 6, 5, 4, 4, 3, 5],
        'I': [7, 7, 8, 8, 6, 8, 7, 6, 5, 4, 5, 7, 8, 6, 7, 5, 6, 3],
    },
)
print(res)
for s, v in res['delivered'].items():
    print('{}: {}'.format(s, v))
