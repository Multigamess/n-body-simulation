from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


WIDTH = 100.0
HEIGHT = 100.0
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


def insert(tree: QuadNode, body: Body):

    if tree.type is NodeType.EXTERNAL:

        if tree.body is None:
            tree.body = body
            return tree

        old_body = tree.body

        new_tree = make_children(tree)

        quad_old = get_quadrant(new_tree, old_body)
        quad_new = get_quadrant(new_tree, body)

        setattr(new_tree, quad_old,
                insert(getattr(new_tree, quad_old), old_body))
        setattr(new_tree, quad_new,
                insert(getattr(new_tree, quad_new), body))

        return new_tree

    else:  # internal node
        quad = get_quadrant(tree, body)
        setattr(tree, quad,
                insert(getattr(tree, quad), body))
        return tree


class QuadTree:

    def __init__(self, width: float, height: float):
        self.root = ExternalNode(.0, .0, width, height, None)

    def insert(self, body: Body):
        self.root = insert(self.root, body)

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


def gravitational_force(other_body: Body, body: Body) -> np.ndarray:
    if other_body == body:
        return np.zeros(2)
    diff = other_body.position - body.position
    return -G * other_body.mass * body.mass / np.linalg.norm(diff)**3 * diff


THETA = 0.5


def node_mass(tree: QuadNode):
    if tree.type is NodeType.INTERNAL:
        return node_mass(tree.top_l) + node_mass(tree.top_r) + node_mass(tree.bot_l) + node_mass(tree.bot_r)
    else:
        return 0 if tree.body is None else tree.body.mass


def compute_force(tree: QuadNode, body: Body):
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
            total_mass = node_mass(tree)
            node_body = Body(position=np.array(
                [mid_x, mid_y]), mass=total_mass, velocity=np.zeros(2))
            return gravitational_force(node_body, body)
        else:
            return compute_force(tree.top_l, body) + compute_force(tree.top_r, body) + compute_force(tree.bot_l, body) + compute_force(tree.bot_r, body)


tree = QuadTree(WIDTH, HEIGHT)
N_BODIES = 1_000
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


def update(frame):
    global bodies

    tree = QuadTree(WIDTH, HEIGHT)
    tree.insert_list(bodies)

    for body in bodies:
        force = compute_force(tree.root, body)
        acceleration = force/body.mass
        body.velocity = body.velocity + DT * acceleration
        body.position = body.position + DT * body.velocity

    xs = [b.position[0] for b in bodies]
    ys = [b.position[1] for b in bodies]
    scatter.set_data(xs, ys)
    return scatter,


anim = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()
