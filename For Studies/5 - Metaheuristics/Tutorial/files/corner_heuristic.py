from framework import Orientation, Problem, Box, Corner, Container
from typing import TypeVar, List, Callable
import argparse
import random as rand


def check_fit(cont: Container, corner: Corner, box: Box) -> bool:
    """True if box (with orientation) fits at particular corner in container"""
    assert(corner in cont.corners)
    # Unimplemented so always return false
    return False

T = TypeVar('T')
OrderFn =  Callable[[List[T]], List[T]]  # Generic ordering function type

def corner_heuristic(prob: Problem,
                     order_boxes: OrderFn[Box]=lambda x: x,
                     order_conts: OrderFn[Container]=lambda x: x,
                     order_corners: OrderFn[Corner]=lambda x: x,
                     order_orients: OrderFn[Orientation]=lambda x: x,
                     ):
    """The corner heuristic algorithm"""
    # Unimplemented
    pass


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
                     order_boxes=lambda x: x,
                     order_conts=lambda x: x,
                     order_corners=lambda x: x,
                     order_orients=lambda x: x,
                     )
    print(prob.objective())
    if not args.no_plot:
        prob.plot(file_name=args.save_plot)
