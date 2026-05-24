import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 10_000

MAX_NODES = 8 * N
WIDTH = 100.0
MIN_CELL_SIZE = 1e-2
EPSILON = 0.005
EPSILON_SQ = EPSILON*EPSILON
THETA = 0.5
THETA_SQ = THETA*THETA
G = 500.0
DT = 0.05

node_x0 = np.zeros(MAX_NODES)
node_y0 = np.zeros(MAX_NODES)
node_size = np.zeros(MAX_NODES)
node_mass = np.zeros(MAX_NODES)
node_bar_x = np.zeros(MAX_NODES)
node_bar_y = np.zeros(MAX_NODES)
# children[k, q] = index of child q (0..3) of node k, -1 if no child
node_children = np.full((MAX_NODES, 4), -1, dtype=np.int64)
# for leaves: which body sits here (-1 with no body)
node_body = np.full(MAX_NODES, -1, dtype=np.int64)
node_is_leaf = np.ones(MAX_NODES, dtype=np.bool_)


def quadrant(px, py, x0, y0, size):
    half = size * 0.5
    right = 1 if px > x0+half else 0
    bottom = 1 if py > y0+half else 0
    return (bottom << 1) | right


def build_tree(pos, mass, N):
    node_children.fill(-1)
    node_body.fill(-1)
    node_is_leaf.fill(True)
    node_mass.fill(0.0)
    node_bar_x.fill(0.0)
    node_bar_y.fill(0.0)

    node_x0[0] = 0.0
    node_y0[0] = 0.0
    node_size[0] = WIDTH
    node_is_leaf[0] = True
    node_body[0] = -1

    n_nodes = 1
    for i in range(N):
        n_nodes = insert_body(i, pos, mass, n_nodes)
    return n_nodes


def insert_body(i, pos, mass, n_nodes):
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
                _init_child(k, q, child)
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

        _init_child(k, qo, c)
        node_body[c] = old
        node_mass[c] = mass[old]
        node_bar_x[c] = pos[old, 0]
        node_bar_y[c] = pos[old, 1]

        node_mass[k] = mass[old]
        node_bar_x[k] = pos[old, 0]
        node_bar_y[k] = pos[old, 1]


def _init_child(parent, q, child):
    half = node_size[parent] * 0.5
    right = q & 1
    bottom = (q >> 1) & 1
    node_x0[child] = node_x0[parent] + right * half
    node_y0[child] = node_y0[parent] + bottom * half
    node_size[child] = half
    node_is_leaf[child] = True
    node_children[child].fill(-1)


def compute_force(i, pos, mass):
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
    return fx, fy


def all_forces(pos, mass):
    F = np.zeros_like(pos)
    for i in range(len(pos)):
        fx, fy = compute_force(i, pos, mass)
        F[i, 0] = fx
        F[i, 1] = fy
    return F


fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, WIDTH)
ax.set_aspect('equal')

scatter = ax.plot([], [], 'ro', markersize=2)[0]

pos = np.random.random((N, 2)) * np.array([WIDTH, WIDTH])
vel = (np.random.random((N, 2)) - 0.5) * 2.0
mass = np.ones(N)


def update(frame):
    n_nodes = build_tree(pos, mass, N)
    forces = all_forces(pos, mass)
    acc = forces / mass[:, np.newaxis]
    vel[:] += DT * acc
    pos[:] += DT * vel

    scatter.set_data(pos[:, 0], pos[:, 1])
    return scatter,


anim = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()
