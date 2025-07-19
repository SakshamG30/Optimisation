import json
from enum import Enum
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec


class Orientation(Enum):
    HORIZONTAL = 1  # rotated so wider than taller
    VERTICAL = 2  # rotated so taller than wider


class Box:
    """A Box

    The only member variable which should be mutated is `orient` to set the
    orientation of the box.
    """
    def __init__(self, id: int, x: int, y: int, weight: int):
        self.id = id
        self.weight = weight
        # Permanently store dimensions in horizontal order
        if x > y:
            self.dims = (x, y)
        else:
            self.dims = (y, x)
        # Start off in horizontal state (you can change this attribute)
        self.orient = Orientation.HORIZONTAL

    def orientated(self) -> Tuple[int, int]:
        """Returns the dimensions of the box, orientated correctly"""
        if self.orient == Orientation.HORIZONTAL:
            return self.dims
        else:
            return (self.dims[1], self.dims[0])


class Corner:
    """A position for a box corner"""
    def __init__(self, x: int, y: int):
        self.pos = (x, y)


class Container:
    """Container for boxes"""
    def __init__(self, id: int, x: int, y: int):
        self.id = id
        self.dims = (x, y)
        # Candidate corner positions within container where boxes can be packed
        self.corners: List[Corner] = [Corner(0, 0)]  # first candidate
        # Packed boxes along with bottom left corner positions
        self.packed: List[Tuple[Corner, Box]] = []

    def pack(self, corner: Corner, box: Box):
        """Pack a box at a particular corner position"""
        # Pack the box
        self.packed.append((corner, box))

        # Remove used corner
        self.corners.remove(corner)

        boxw, boxh = box.orientated()
        # Add new candidate corner at top left corner of newly packed box
        self.corners.append(Corner(corner.pos[0], corner.pos[1] + boxh))
        # Add new candidate corner at bottom right corner of newly pack box
        self.corners.append(Corner(corner.pos[0] + boxw, corner.pos[1]))

    def unpack_all(self) -> List[Box]:
        """Unpack all boxes, returning them"""
        removed_boxes = [b for (_, b) in self.packed]
        self.packed = []
        self.corners = [Corner(0, 0)]
        return removed_boxes


class Problem:
    """The problem instance and solution state mixed together

    Boxes should be moved from unpacked list into containers, and back to
    unpacked list again if container gets unpacked.
    """

    def __init__(self, filename: str):
        data = json.load(open(filename))
        self.conts = [Container(i, c["x"], c["y"])
                      for i, c in enumerate(data["containers"])]
        self.unpacked = [Box(i, c["x"], c["y"], c["w"])
                         for i, c in enumerate(data["boxes"])]

    def objective(self) -> int:
        """Total weight of boxes packed into containers"""
        tot = 0
        for cont in self.conts:
            for _, box in cont.packed:
                tot += box.weight
        return tot

    def plot(self, file_name: Optional[str]=None):
        n_cols = 1000
        n_rows = 2
        n_conts = len(self.conts)
        fig = plt.figure(figsize=(8 * n_conts, 10))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        cmap = cm.plasma
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Plot containers
        g = gridspec.GridSpec(n_rows, n_cols, height_ratios=[3, 1])
        for cont in self.conts:
            col_start = int((cont.id * n_cols) / float(n_conts))
            col_end = int((cont.id + 0.9) * n_cols / float(n_conts))
            ax = fig.add_subplot(g[0, col_start:col_end])

            # Plot packed boxes
            for corner, box in cont.packed:
                ax.add_patch(matplotlib.patches.Rectangle(
                    corner.pos,
                    box.orientated()[0],
                    box.orientated()[1],
                    edgecolor="black",
                    facecolor=m.to_rgba(box.weight)))

                plt.text(corner.pos[0] + 0.5 * box.orientated()[0],
                         corner.pos[1] + 0.5 * box.orientated()[1],
                         str(box.id), ha="center")

            plt.xlim([0, cont.dims[0]])
            plt.ylim([0, cont.dims[1]])

        # Plot unpacked boxes
        ax = fig.add_subplot(g[1, 0:n_cols])
        ax.axis('off')
        x_pos = 0
        max_y = 1
        if len(self.unpacked) > 0:
            for box in self.unpacked:
                box_dims = box.orientated()
                ax.add_patch(matplotlib.patches.Rectangle(
                    (x_pos, 0),
                    box_dims[0],
                    box_dims[1],
                    edgecolor="black",
                    facecolor=m.to_rgba(box.weight)))

                plt.text(x_pos + 0.5 * box_dims[0], 0.5 * box_dims[1],
                         str(box.id), ha="center")

                x_pos += box_dims[0] + 1
                max_y = max(max_y, box_dims[1])

            plt.xlim([0, x_pos])
            plt.ylim([0, max_y * 2])

        if file_name is not None:
            fig.savefig(file_name)
        plt.show()
