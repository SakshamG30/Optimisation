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

    m += 0

    m.solve()
    print(f'Status: {pulp.LpStatus[m.status]}')
    print(f'Cost: {m.objective.value()}')
    print('Generation:')
    print([0. for _ in gens])


if __name__ == "__main__":
    data = json.load(open('pglib_opf_case14_ieee__api-ldc.json'))
    # data = json.load(open('pglib_opf_case1951_rte__api-ldc.json'))
    prb = OpfProblem.from_json(data)
    solve_opf(prb)
