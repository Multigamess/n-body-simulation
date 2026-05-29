from numba import njit, prange
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time


@njit
def quadrant(px, py, x0, y0, size):
    half = size * 0.5
    right = 1 if px > x0+half else 0
    bottom = 1 if py > y0+half else 0
    return (bottom << 1) | right


@njit
def _init_child(parent, q, child,
                node_x0, node_y0, node_size, node_is_leaf, node_children):
    half = node_size[parent] * 0.5
    right = q & 1
    bottom = (q >> 1) & 1
    node_x0[child] = node_x0[parent] + right * half
    node_y0[child] = node_y0[parent] + bottom * half
    node_size[child] = half
    node_is_leaf[child] = True
    for q2 in range(4):
        node_children[child, q2] = -1


@njit
def insert_body(i, pos, mass, n_nodes,
                MIN_CELL_SIZE,
                node_x0, node_y0, node_size, node_mass,
                node_bar_x, node_bar_y, node_children, node_body, node_is_leaf):
    px, py = pos[i, 0], pos[i, 1]
    k = 0

    while True:
        if not node_is_leaf[k]:
            # internal -> updatee barycenter
            m_new = node_mass[k]+mass[i]
            node_bar_x[k] = (node_mass[k]*node_bar_x[k]+mass[i]*px)/m_new
            node_bar_y[k] = (node_mass[k]*node_bar_y[k]+mass[i]*py)/m_new
            node_mass[k] = m_new
            q = quadrant(px, py, node_x0[k], node_y0[k], node_size[k])
            child = node_children[k, q]

            if child == -1:
                child = n_nodes
                n_nodes += 1
                node_children[k, q] = child
                _init_child(k, q, child,
                            node_x0, node_y0, node_size, node_is_leaf, node_children)
                node_body[child] = i
                node_mass[child] = mass[i]
                node_bar_x[child] = px
                node_bar_y[child] = py

                return n_nodes
            k = child
            continue

        # leaf
        if node_body[k] == -1:
            node_body[k] = i
            node_mass[k] = mass[i]
            node_bar_x[k] = px
            node_bar_y[k] = py
            return n_nodes

        if node_size[k] < MIN_CELL_SIZE:
            node_mass[k] += mass[i]
            return n_nodes

        # leaf becomes internal, we subdivide
        old = node_body[k]
        node_is_leaf[k] = False
        node_body[k] = -1
        # place the old body as a child
        qo = quadrant(pos[old, 0], pos[old, 1],
                      node_x0[k], node_y0[k], node_size[k])
        c = n_nodes
        n_nodes += 1
        node_children[k, qo] = c

        _init_child(k, qo, c,
                    node_x0, node_y0, node_size, node_is_leaf, node_children)
        node_body[c] = old
        node_mass[c] = mass[old]
        node_bar_x[c] = pos[old, 0]
        node_bar_y[c] = pos[old, 1]

        node_mass[k] = mass[old]
        node_bar_x[k] = pos[old, 0]
        node_bar_y[k] = pos[old, 1]


@njit
def build_tree(pos, mass, N,
               MIN_CELL_SIZE, WIDTH,
               node_x0, node_y0, node_size, node_mass,
               node_bar_x, node_bar_y, node_children, node_body, node_is_leaf):
    for k in range(node_mass.shape[0]):
        node_mass[k] = 0.0
        node_bar_x[k] = 0.0
        node_bar_y[k] = 0.0
        node_body[k] = -1
        node_is_leaf[k] = True
        for q in range(4):
            node_children[k, q] = -1

    node_x0[0] = 0.0
    node_y0[0] = 0.0
    node_size[0] = WIDTH
    node_is_leaf[0] = True
    node_body[0] = -1

    n_nodes = 1
    for i in range(N):
        n_nodes = insert_body(i, pos, mass, n_nodes,
                              MIN_CELL_SIZE,
                              node_x0, node_y0, node_size, node_mass,
                              node_bar_x, node_bar_y, node_children, node_body, node_is_leaf)
    return n_nodes


@njit
def all_forces(pos, mass,
               EPSILON_SQ, THETA_SQ, G,
               node_size, node_mass, node_bar_x, node_bar_y,
               node_children, node_body, node_is_leaf):
    n = pos.shape[0]
    F = np.zeros((n, 2))
    for i in prange(n):
        px, py = pos[i, 0], pos[i, 1]
        fx = fy = 0.0
        stack = np.empty(64, dtype=np.int64)  # depth bound
        sp = 0
        stack[sp] = 0
        sp += 1

        while sp > 0:
            sp -= 1
            k = stack[sp]

            if node_is_leaf[k]:
                b = node_body[k]
                if b != -1 and b != i:
                    dx = pos[b, 0]-px
                    dy = pos[b, 1]-py
                    d2 = dx*dx + dy*dy
                    soft = d2 + EPSILON_SQ
                    f = G * mass[b] * mass[i] / (soft * np.sqrt(soft))
                    fx += f*dx
                    fy += f*dy

            else:
                dx = node_bar_x[k] - px
                dy = node_bar_y[k]-py
                d2 = dx*dx + dy*dy

                if node_size[k]*node_size[k] < THETA_SQ * d2:
                    soft = d2 + EPSILON_SQ
                    f = G * node_mass[k] * mass[i] / (soft*np.sqrt(soft))
                    fx += f*dx
                    fy += f*dy

                else:
                    for q in range(4):
                        c = node_children[k, q]
                        if c != -1:
                            stack[sp] = c
                            sp += 1
        F[i, 0] = fx
        F[i, 1] = fy
    return F


class BarnesHut:

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray,
                 WIDTH: float, MIN_CELL_SIZE: float, EPSILON: float, THETA: float,
                 G: float, DT: float) -> None:

        self.initial_positions = positions
        self.positions = positions.copy()
        self.initial_velocities = velocities
        self.velocities = velocities.copy()
        self.masses = masses

        self.N = positions.shape[0]
        self.MAX_NODES = 8 * self.N
        self.WIDTH = WIDTH
        self.MIN_CELL_SIZE = MIN_CELL_SIZE
        self.EPSILON = EPSILON
        self.EPSILON_SQ = EPSILON*EPSILON
        self.THETA = THETA
        self.THETA_SQ = THETA*THETA
        self.G = G
        self.DT = DT

        self.node_x0 = np.zeros(self.MAX_NODES)
        self.node_y0 = np.zeros(self.MAX_NODES)
        self.node_size = np.zeros(self.MAX_NODES)
        self.node_mass = np.zeros(self.MAX_NODES)
        self.node_bar_x = np.zeros(self.MAX_NODES)
        self.node_bar_y = np.zeros(self.MAX_NODES)
        # children[k, q] = index of child q (0..3) of node k, -1 if no child
        self.node_children = np.full((self.MAX_NODES, 4), -1, dtype=np.int64)
        # for leaves: which body sits here (-1 with no body)
        self.node_body = np.full(self.MAX_NODES, -1, dtype=np.int64)
        self.node_is_leaf = np.ones(self.MAX_NODES, dtype=np.bool_)

    def build_bh_tree(self):
        return build_tree(self.positions, self.masses, self.N,
                          self.MIN_CELL_SIZE, self.WIDTH,
                          self.node_x0, self.node_y0, self.node_size, self.node_mass,
                          self.node_bar_x, self.node_bar_y, self.node_children, self.node_body, self.node_is_leaf)

    def next_step(self):
        n_nodes = self.build_bh_tree()
        forces = all_forces(self.positions, self.masses,
                            self.EPSILON_SQ, self.THETA_SQ, self.G,
                            self.node_size, self.node_mass, self.node_bar_x, self.node_bar_y,
                            self.node_children, self.node_body, self.node_is_leaf)
        acc = forces / self.masses[:, np.newaxis]
        self.velocities[:] += self.DT * acc
        self.positions[:] += self.DT * self.velocities

    def simulate(self, number_of_steps: int):
        all_positions = np.zeros((number_of_steps, self.N, 2))
        all_velocities = np.zeros((number_of_steps, self.N, 2))
        all_times = np.zeros(number_of_steps)
        print("== Barnes-Hut ==")
        print("Simulation starting...")
        start_time = time.time()
        for i in range(number_of_steps):
            if i % 10 == 0:
                print(
                    f"Current step : {i} - Time elapsed : {(time.time()-start_time):.2f} s")
            all_positions[i] = self.positions.copy()
            all_velocities[i] = self.velocities.copy()
            all_times[i] = i*self.DT
            self.next_step()

        print("Simulation finished...")
        return (all_positions, all_velocities, all_times)


def plot(all_positions: np.ndarray, all_velocities: np.ndarray, all_times: np.ndarray, WIDTH: float, number_of_steps: int):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, WIDTH)
    ax.set_aspect('equal')

    time_text = ax.text(0.05, 0.95, '', horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes)

    scatter = ax.plot([], [], 'ro', markersize=2)[0]

    def update(frame):
        i = (frame-1) % number_of_steps
        current_time = all_times[i]
        scatter.set_data(all_positions[i, :, 0],
                         all_positions[i, :, 1])
        time_text.set_text(f'time = {current_time:.2f}s')
        return scatter,

    anim = FuncAnimation(fig, update, interval=30, blit=False)
    plt.show()

# initial positions generation


def circle(N: int, cx: float, cy: float, radius: float, omega: float):
    r = np.random.random(N) * radius
    phi = np.random.random(N) * 2 * np.pi
    positions = np.column_stack((r * np.cos(phi)+cx, r * np.sin(phi)+cy))

    positions_3d = np.column_stack((positions, np.zeros(N)))
    omega_vec = np.array([0.0, 0.0, omega])

    velocities_3d = np.cross(omega_vec, positions_3d)
    velocities = velocities_3d[:, :2]

    return positions, velocities


if __name__ == "__main__":

    WIDTH = 100
    MIN_CELL_SIZE = 1e-4
    N = 10_000

    # positions = np.random.random((N, 2)) * np.array([WIDTH, WIDTH])
    # velocities = (np.random.random((N, 2)) - 0.5) * 2.0

    positions, velocities = circle(N, WIDTH/2, WIDTH/2, WIDTH/4, 0.01)
    masses = np.ones(N)
    EPSILON = 1.0
    THETA = 0.5
    G = 1.0
    DT = 0.05

    number_of_steps = 1000

    barnes_hut = BarnesHut(positions,  velocities, masses,
                           WIDTH, MIN_CELL_SIZE, EPSILON, THETA, G, DT)

    all_positions, all_velocities, all_times = barnes_hut.simulate(
        number_of_steps)

    plot(all_positions, all_velocities, all_times, WIDTH, number_of_steps)
