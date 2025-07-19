from framework import Orientation, Problem, Box, Corner, Container
from typing import TypeVar, List, Tuple, Callable
import argparse
import random as rand


def contained_inside(p1: Tuple[int, int],
                     size1: Tuple[int, int],
                     p2: Tuple[int, int],
                     size2: Tuple[int, int],
                     ) -> bool:
    # 2 in 1
    l1 = p1[0]
    r1 = p1[0] + size1[0]
    l2 = p2[0]
    r2 = p2[0] + size2[0]
    b1 = p1[1]
    t1 = p1[1] + size1[1]
    b2 = p2[1]
    t2 = p2[1] + size2[1]

    return l2 >= l1 and r2 <= r1 and b2 >= b1 and t2 <= t1


def are_separate(p1: Tuple[int, int],
                 size1: Tuple[int, int],
                 p2: Tuple[int, int],
                 size2: Tuple[int, int],
                 ) -> bool:
    l1 = p1[0]
    r1 = p1[0] + size1[0]
    l2 = p2[0]
    r2 = p2[0] + size2[0]
    b1 = p1[1]
    t1 = p1[1] + size1[1]
    b2 = p2[1]
    t2 = p2[1] + size2[1]

    return l1 >= r2 or l2 >= r1 or b1 >= t2 or b2 >= t1


def check_fit(cont: Container, corner: Corner, box: Box) -> bool:
    """True if box (with orientation) fits at particular corner in container"""
    assert(corner in cont.corners)
    # First check if it fits in container at corner
    if not contained_inside((0, 0),
                            cont.dims,
                            corner.pos,
                            box.orientated()):
        return False
    # Check overlap with other boxes in container
    for (c, b) in cont.packed:
        if not are_separate(corner.pos,
                            box.orientated(),
                            c.pos,
                            b.orientated()):
            return False
    return True


T = TypeVar('T')
OrderFn =  Callable[[List[T]], List[T]]  # Generic ordering function type

def corner_heuristic(prob: Problem,
                     order_boxes: OrderFn[Box]=lambda x: x,
                     order_conts: OrderFn[Container]=lambda x: x,
                     order_corners: OrderFn[Corner]=lambda x: x,
                     order_orients: OrderFn[Orientation]=lambda x: x,
                     ):
    """The corner heuristic algorithm"""
    unpacked = []
    for box in order_boxes(prob.unpacked):
        packed = False
        for cont in order_conts(prob.conts):
            for corner in order_corners(cont.corners):
                for orient in order_orients([Orientation.HORIZONTAL,
                                             Orientation.VERTICAL]):
                    box.orient = orient
                    if check_fit(cont, corner, box):
                        cont.pack(corner, box)
                        packed = True
                        break
                if packed:
                    break
            if packed:
                break
        if not packed:
            box.orient = Orientation.HORIZONTAL
            unpacked.append(box)
    prob.unpacked = unpacked


if __name__ == "__main__":
    par = argparse.ArgumentParser("2D Packing Corner Heuristic Solver")
    par.add_argument("file", help="json instance file")
    par.add_argument("--save-plot", default=None,
                     help="save plot to file")
    par.add_argument("--no-plot", action='store_true',
                     help="don't plot solution")

    args = par.parse_args()

    rand.seed(0)

    prob = Problem(args.file)
    # Play around with the ordering functions here in Q1.3:
    corner_heuristic(prob,
                     order_boxes=lambda x: sorted(x, reverse=True,
                                                  key=lambda y: y.weight),
                     #order_conts=lambda x: rand.sample(x, len(x)),
                     #order_corners=lambda x: rand.sample(x, len(x)),
                     #order_orients=lambda x: rand.sample(x, len(x)),
                     )
    print(prob.objective())
    if not args.no_plot:
        prob.plot(file_name=args.save_plot)
