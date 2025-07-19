from __future__ import annotations
import pulp
from pulp import LpVariable as Var
from pulp import lpSum
import json
from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    id: int  # indentifier
    load: float  # load at node (power)
    

@dataclass
class Line:
    node_fr: int  # from Node id
    node_to: int  # to Node id
    p_max: float  # maximum power on line (power)
    b: float  # line susceptance (power / radian)


@dataclass
class Generator:
    node: int  # connected Node id
    p_max: float  # maximum generator power (power)
    price: float  # price of generation ($ / power / hour)


@dataclass
class OpfProblem:
    nodes: List[Node]
    lines: List[Line]
    gens: List[Generator]

    @classmethod
    def from_json(cls, data) -> OpfProblem:
        """Parse json data"""
        s_base = data['S_base']
        nodes = [Node(id=v['id'],
                      load=v['L'] / s_base)
                 for v in data['nodes']]
        lines = [Line(node_fr=v['nodes'][0],
                      node_to=v['nodes'][1],
                      p_max=v['P_max'] / s_base,
                      b=v['b'])
                 for v in data['lines']]
        gens = [Generator(node=v['node'],
                          p_max=v['G_max'] / s_base,
                          price=v['price'] * s_base)
                for v in data['gens']]
        return OpfProblem(nodes, lines, gens)


def solve_opf(prb: OpfProblem):
    m = pulp.LpProblem()

    nodes = prb.nodes
    lines = prb.lines
    gens = prb.gens

    # Phase angles
    ths = {node.id: Var(f'th_{node.id}') for node in nodes}
    # Line powers
    ps = [Var(f'p_{i}', -line.p_max, line.p_max) for i, line in enumerate(lines)]
    # Generator powers
    gs = [Var(f'g_{i}', 0, gen.p_max) for i, gen in enumerate(gens)]

    # Node 1 is the reference node
    m += ths[1] == 0

    # Power flow relation
    for line, p in zip(lines, ps):
        m += p == - line.b * (ths[line.node_fr] - ths[line.node_to])

    # Collect together the line power flowing in and out of each node
    p_in = {node.id: [] for node in nodes}
    p_out = {node.id: [] for node in nodes}
    for line, p in zip(lines, ps):
        p_in[line.node_to].append(p)
        p_out[line.node_fr].append(p)

    # Collect the generators injecting into each node
    p_gen = {node.id: [] for node in nodes}
    for gen, g in zip(gens, gs):
        p_gen[gen.node].append(g)

    # Conservation of power at each node (Kirchoff's junction law)
    for node in nodes:
        nid = node.id
        m += (lpSum(p_in[nid]) - lpSum(p_out[nid]) + lpSum(p_gen[nid])
              - node.load) == 0

    m += pulp.lpDot(gs, [gen.price for gen in gens])

    m.solve()
    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Cost: {m.objective.value()}')
    print('Generation:')
    print([g.value() for g in gs])


if __name__ == "__main__":
    data = json.load(open('../files/pglib_opf_case14_ieee__api-ldc.json'))
    #data = json.load(open('../files/pglib_opf_case1951_rte__api-ldc.json'))
    prb = OpfProblem.from_json(data)
    solve_opf(prb)

