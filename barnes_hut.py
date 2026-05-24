from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys


WIDTH = 100.0
HEIGHT = 100.0
MIN_CELL_SIZE = 1e-2
G = 1.0


class Body:

    _id_counter = 0

    def __init__(self,  position: np.ndarray, mass: float, velocity: np.ndarray):
        self.id = Body._id_counter
        Body._id_counter += 1

        self.position = position
        self.mass = mass
        self.velocity = velocity

    def __eq__(self, other: 'Body'):
        return other.id == self.id


def random_bodies(N: int):
    coords = np.random.random((N, 2)) * np.array([WIDTH, HEIGHT])
    bodies = [Body(coords[i], 1.0, np.zeros(2)) for i in range(N)]
    return bodies


class NodeType(Enum):
    EXTERNAL = 1
    INTERNAL = 2


class QuadNode():

    def __init__(self,
                 node_type: NodeType,
                 x0: float, y0: float, width: float, height: float):
        self.type = node_type
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height


class InternalNode(QuadNode):

    def __init__(self,
                 x0: float, y0: float, width: float, height: float,
                 top_l: 'QuadNode', top_r: 'QuadNode', bot_l: 'QuadNode', bot_r: 'QuadNode'):
        super().__init__(NodeType.INTERNAL, x0, y0, width, height)
        self.top_l = top_l
        self.top_r = top_r
        self.bot_l = bot_l
        self.bot_r = bot_r
        self.total_mass = 0.0
        self.bar_x = 0.0
        self.bar_y = 0.0


class ExternalNode(QuadNode):
    def __init__(self,
                 x0: float, y0: float, width: float, height: float,
                 body: np.ndarray | None):
        super().__init__(NodeType.EXTERNAL, x0, y0, width, height)
        self.body = body


def get_quadrant(node: QuadNode, body: Body) -> str:
    mid_x = node.x0 + node.width/2
    mid_y = node.y0 + node.height/2

    if body.position[0] <= mid_x:
        # left
        if body.position[1] <= mid_y:
            # top
            return 'top_l'
        else:
            # bot
            return 'bot_l'
    else:
        # right
        if body.position[1] <= mid_y:
            # top
            return 'top_r'
        else:
            # bot
            return 'bot_r'


def make_children(node: ExternalNode) -> InternalNode:
    mid_x = node.x0 + node.width/2
    mid_y = node.y0 + node.height/2

    w2 = node.width/2
    h2 = node.height/2

    return InternalNode(node.x0, node.y0, node.width, node.height,
                        top_l=ExternalNode(node.x0, node.y0, w2, h2, None),
                        top_r=ExternalNode(mid_x, node.y0, w2, h2, None),
                        bot_l=ExternalNode(node.x0, mid_y, w2, h2, None),
                        bot_r=ExternalNode(mid_x, mid_y, w2, h2, None))


def insert_recursive(tree: QuadNode, body: Body):

    if tree.type is NodeType.EXTERNAL:

        if tree.body is None:
            tree.body = body
            return tree

        old_body = tree.body

        new_tree = make_children(tree)

        quad_old = get_quadrant(new_tree, old_body)
        quad_new = get_quadrant(new_tree, body)

        setattr(new_tree, quad_old,
                insert_recursive(getattr(new_tree, quad_old), old_body))
        setattr(new_tree, quad_new,
                insert_recursive(getattr(new_tree, quad_new), body))

        new_tree.total_mass += old_body.mass + body.mass

        return new_tree

    else:  # internal node
        quad = get_quadrant(tree, body)
        tree.total_mass += body.mass
        setattr(tree, quad,
                insert_recursive(getattr(tree, quad), body))
        return tree


def insert_iterative(tree: QuadNode, body: Body):
    node = tree
    parent = None
    parent_attr = None

    while True:
        if node.type is NodeType.INTERNAL:
            total_mass_new = node.total_mass + body.mass
            node.bar_x = (node.total_mass * node.bar_x +
                          body.mass * body.position[0])/total_mass_new
            node.bar_y = (node.total_mass * node.bar_y +
                          body.mass * body.position[1])/total_mass_new
            node.total_mass = total_mass_new

            parent = node
            parent_attr = get_quadrant(parent, body)
            node = getattr(parent, parent_attr)
            continue

        # external node
        if node.body is None:
            node.body = body
            return tree

        # leaf is not empty, we need to subdivide
        # prevent infinite recursion here with a minimum cell size
        if node.width < MIN_CELL_SIZE or node.height < MIN_CELL_SIZE:
            # we combine the 2 masses
            node.body.mass += body.mass
            return tree

        old_body = node.body

        new_node = make_children(node)

        if parent is None:
            tree = new_node
        else:
            setattr(parent, parent_attr, new_node)

        new_node.total_mass += old_body.mass
        new_node.bar_x = old_body.position[0]
        new_node.bar_y = old_body.position[1]
        q_old = get_quadrant(new_node, old_body)
        getattr(new_node, q_old).body = old_body

        node = new_node

        parent = parent


class QuadTree:

    def __init__(self, width: float, height: float):
        self.root = ExternalNode(.0, .0, width, height, None)

    def insert(self, body: Body):
        self.root = insert_iterative(self.root, body)

    def insert_list(self, bodies: list[type[Body]]):
        for b in bodies:
            self.insert(b)


def plot_quadtree(tree: QuadNode, ax: plt.Axes):

    if tree.type is NodeType.EXTERNAL:
        if tree.body is not None:
            body = tree.body
            ax.plot(body.position[0], body.position[1], 'ro', zorder=3)
        return
    else:  # internal node

        mid_x = tree.x0 + tree.width/2
        mid_y = tree.y0 + tree.height/2
        ax.plot([mid_x, mid_x], [tree.y0, tree.y0 +
                tree.height], 'green', linewidth=1)
        ax.plot([tree.x0, tree.x0+tree.width],
                [mid_y, mid_y], 'green', linewidth=1)

        plot_quadtree(tree.top_l, ax)
        plot_quadtree(tree.top_r, ax)
        plot_quadtree(tree.bot_l, ax)
        plot_quadtree(tree.bot_r, ax)


EPSILON = 1.0
EPSILON_SQ = EPSILON*EPSILON


def gravitational_force(other_body: Body, body: Body) -> np.ndarray:
    if other_body == body:
        return np.zeros(2)
    diff = other_body.position - body.position
    return G * other_body.mass * body.mass / (np.linalg.norm(diff)+EPSILON)**3 * diff


THETA = 0.5
THETA_SQ = THETA * THETA


def compute_force_recursive(tree: QuadNode, body: Body):
    if tree.type is NodeType.EXTERNAL:
        other_body = tree.body

        return np.zeros(2) if other_body is None else gravitational_force(other_body, body)
    else:

        s = tree.width
        mid_x = tree.x0 + tree.width/2
        mid_y = tree.y0 + tree.height/2
        d = np.sqrt((mid_x - body.position[0])
                    ** 2+(mid_y - body.position[1])**2)

        if s/d < THETA:
            # treat as external node
            total_mass = tree.total_mass
            node_body = Body(position=np.array(
                [mid_x, mid_y]), mass=total_mass, velocity=np.zeros(2))
            return gravitational_force(node_body, body)
        else:
            return compute_force_recursive(tree.top_l, body) + compute_force_recursive(tree.top_r, body) + compute_force_recursive(tree.bot_l, body) + compute_force_recursive(tree.bot_r, body)


def compute_force_iterative(tree: QuadNode, body: Body):
    px, py = body.position[0], body.position[1]
    pmass = body.mass
    fx, fy = 0.0, 0.0

    stack = [tree]
    while stack:
        node = stack.pop()

        if node.type is NodeType.EXTERNAL:
            node_body = node.body
            if node_body is not None and node_body != body:
                dx = node_body.position[0] - px
                dy = node_body.position[1] - py
                d2 = dx*dx + dy*dy
                soft = d2 + EPSILON_SQ
                factor = G * node_body.mass * pmass / (soft * np.sqrt(soft))
                fx += factor * dx
                fy += factor * dy

        else:

            dx = node.bar_x - px
            dy = node.bar_y - py
            d2 = dx*dx + dy*dy

            if node.width * node.width < THETA_SQ * d2:
                soft = d2 + EPSILON_SQ
                factor = G * node.total_mass * pmass / (soft * np.sqrt(soft))
                fx += factor * dx
                fy += factor * dy
            else:
                stack.append(node.top_l)
                stack.append(node.top_r)
                stack.append(node.bot_l)
                stack.append(node.bot_r)

    return (fx, fy)


tree = QuadTree(WIDTH, HEIGHT)
N_BODIES = 1000
bodies = random_bodies(N_BODIES)
tree.insert_list(bodies)

fig, ax = plt.subplots()

ax.add_patch(
    patches.Rectangle(
        xy=(0.0, 0.0),
        width=WIDTH, height=HEIGHT, linewidth=1,
        color='green', fill=False))
ax.axis('equal')

plot_quadtree(tree.root, ax)

plt.show()

DT = 0.1

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')

scatter = ax.plot([], [], 'ro', markersize=2)[0]


def compute_all_forces(bodies: list[type[Body]], tree: QuadNode):
    forces = np.zeros((len(bodies), 2))
    for i, body in enumerate(bodies):
        forces[i] = compute_force_iterative(tree, body)
    return forces


def update(frame):
    global bodies

    tree = QuadTree(WIDTH, HEIGHT)
    tree.insert_list(bodies)

    positions = np.array([b.position for b in bodies])
    velocities = np.array([b.velocity for b in bodies])
    masses = np.array([b.mass for b in bodies])

    forces = compute_all_forces(bodies, tree.root)
    accelerations = forces / masses[:, np.newaxis]

    velocities += DT * accelerations
    positions += DT * velocities

    for i, body in enumerate(bodies):
        body.velocity = velocities[i]
        body.position = positions[i]

    scatter.set_data(positions[:, 0], positions[:, 1])
    return scatter,


anim = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()
