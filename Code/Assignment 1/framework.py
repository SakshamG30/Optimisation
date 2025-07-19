# Copyright subsists in all the material on this repository and vests in the ANU
# (or other parties). Any copying of the material from this site other than for
# the purpose of working on this assignment is a breach of copyright.
#
# In practice, this means that you are not allowed to make any of the existing
# material in this repository nor your assignment available to others (except to
# the course lecturers and tutors) at any time before, during, or after the
# course. In particular, you must keep any clone of this repository and any
# extension of it private at all times.

"""Invaders problem instance classes and plotting and animation.

The Problem, Laser and UFO classes are the only things you are probably
interested in.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def distance(a, b):
    """Manhattan distance between two 2D points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Problem:
    """Invaders problem instance."""

    def __init__(self, filename):
        """Load in from JSON file."""
        data = json.load(open(filename))
        self.lasers = [Laser(l["position"], l["range"], l["base_damage"],
                             l["falloff"], l["cooldown"], l["size"],
                             l["speed"])
                       for l in data["lasers"]]
        self.ufos = [UFO(u["health"], u["trajectory"]) for u in data["ufos"]]
        self.timesteps = 0  # total number of timesteps
        # The timesteps of the game is given by the maximum length of any UFO
        # trajectory.
        for u in self.ufos:
            self.timesteps = max(len(u.trajectory), self.timesteps)


class UFO:
    """An Unidentified Flying Object that we want to destroy."""

    def __init__(self, health, trajectory):
        self.health = health  # starting "health" of the UFO
        self.trajectory = trajectory  # position of UFO over time

    def position(self, time):
        """Position of UFO for a give time step."""
        if time >= len(self.trajectory):  # if time step beyond end, repeat last
            return self.trajectory[-1]
        return self.trajectory[time]


class Laser:
    def __init__(self, pos, range, base_damage, falloff, cooldown, size,
                 speed):
        self.pos = pos  # fixed position of laser (starting pos for ultra-hard)
        self.range = range  # range of laser as a [min, max] distance pair
        self.base_damage = base_damage  # base damage (at range 0 if possible)
        self.falloff = falloff  # falloff of damage in per length
        self.cooldown = cooldown  # cooldown / retargetting time
        self.size = size  # physical size of laser (for disabling calculations)
        self.speed = speed  # max speed of laser (for ultra-hard calculations)

    def position(self):
        return self.pos

    def damage(self, ufo_pos, laser_pos=None, ignore_min_range=False, tol=0.0):
        """Returns damage that is dealt to a UFO at a given position.

        Accounts for range of laser and if target has reached the bottom.
        Defaults to using fixed laser position, but gives option to calculate
        it for a different laser position (for Ultra-Hard mode).
        """
        laser_pos = self.pos if laser_pos is None else laser_pos
        dist = distance(laser_pos, ufo_pos)
        min_range = 0 if ignore_min_range else self.range[0]
        if ufo_pos[1] <= 0:  # can't do any damage when at or below y = 0
            return 0
        elif min_range - tol <= dist <= self.range[1] + tol:
            # in laser range
            return self.base_damage + self.falloff * dist
        return 0


def when_destroyed(prob, targeting, movement=None):
    """Time that each UFO is destroyed (None if not destroyed).

    Takes the targeting information as given (doesn't check if feasible).
    Can take movement of laser which is only applicable for the Ultra-Hard
    mode.

    Has some tolerances to allow for some numerical rounding errors.
    """
    healths = [ufo.health for ufo in prob.ufos]
    when_destroyed = [None for _ in prob.ufos]
    ignore_min_range = False if movement is None else True  # for Ultra-Hard
    for t in range(prob.timesteps):
        for l, (laser, targets) in enumerate(zip(prob.lasers, targeting)):
            u = targets[t]
            if u is None:
                continue
            laser_pos = None if movement is None else movement[l][t]
            healths[u] -= laser.damage(prob.ufos[u].position(t),
                                       laser_pos=laser_pos,
                                       ignore_min_range=ignore_min_range,
                                       tol=1e-6)
            if healths[u] <= 1e-6 and when_destroyed[u] is None:
                when_destroyed[u] = t

    return when_destroyed


def add_shot(time, ax, ufos, destroyed, laser_pos, targets, shot_line):
    j = targets[time]
    if j is not None and (destroyed[j] is None or destroyed[j] >= time):
        pos_j = ufos[j].position(time)
        pos_i = laser_pos
        xs = [pos_i[0], pos_j[0]]
        ys = [pos_i[1], pos_j[1]]
        if shot_line is None:
            shot_line = ax.plot(xs, ys, color="red")[0]
        else:
            shot_line.set_data(xs, ys)
    elif shot_line is not None:
        shot_line.remove()
        shot_line = None
    return shot_line


def plot_till(time, ax, lines, prob, destroyed, targeting, movement,
              all_shots):
    time = min(time, prob.timesteps - 1)
    for line, ufo, destroy in zip(lines["ufo"], prob.ufos, destroyed):
        trajectory = ufo.trajectory[:time+1]
        if destroy is not None:
            trajectory = trajectory[:destroy+1]
        if len(trajectory) > 0:
            xs, ys = zip(*trajectory)
            line.set_data(xs, ys)
        else:
            line.set_data([], [])

    if movement is not None:
        for line, laser, laser_mov in zip(lines["laser"], prob.lasers,
                                          movement):
            mov = [laser.pos] + laser_mov[:time+1]
            xs, ys = zip(*mov)
            line.set_data(xs, ys)

    for l, targets in enumerate(targeting):
        if all_shots:
            for t in range(time):
                laser_pos = (prob.lasers[l].pos
                             if movement is None else movement[l][t])
                add_shot(t, ax, prob.ufos, destroyed, laser_pos, targets,
                         None)
        else:
            laser_pos = (prob.lasers[l].pos
                         if movement is None else movement[l][time])
            lines["shot"][l] = add_shot(time, ax, prob.ufos, destroyed,
                                        laser_pos, targets, lines["shot"][l])


def plot_init(prob):
    fig, ax1 = plt.subplots(1, 1)

    laser_lines = [ax1.plot([laser.pos[0]], [laser.pos[1]], marker="o")[0]
                   for laser in prob.lasers]

    ufo_lines = [ax1.plot([], [], marker="x", ls="--")[0] for _ in prob.ufos]

    ax1.set_ylim([0.0, 100.0])
    ax1.set_xlim([-50.0, 50.0])
    plt.gca().set_aspect('equal', adjustable='box')

    shot_lines = [None for _ in prob.lasers]

    return fig, ax1, {"ufo": ufo_lines,
                      "laser": laser_lines,
                      "shot": shot_lines}


def plot(prob, targeting, fn, movement=None):
    """Plot the solution.

    "targeting" is a list of list of ints that represent for each laser and
    each time step the targeted UFO index or None if no target.
    """
    _, ax, lines = plot_init(prob)
    destroyed = when_destroyed(prob, targeting, movement)
    plot_till(prob.timesteps, ax, lines, prob, destroyed, targeting, movement,
              True)
    plt.savefig(fn)


def animate(prob, targeting, fn, movement=None):
    """Animate the solution.

    "targeting" is a list of list of ints that represent for each laser and
    each time step the targeted UFO index or None if no target.
    """
    fig, ax, lines = plot_init(prob)
    destroyed = when_destroyed(prob, targeting, movement)
    anim = FuncAnimation(fig, plot_till, frames=prob.timesteps + 4,
                         fargs=(ax, lines, prob, destroyed, targeting,
                                movement, False),
                         interval=400, blit=False)
    anim.save(fn)
