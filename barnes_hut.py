from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


WIDTH = 100.0
HEIGHT = 100.0


def random_bodies(N: int):
    return np.random.random((N, 2)) * np.array([WIDTH, HEIGHT])


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


def get_quadrant(node: QuadNode, body: np.ndarray) -> str:
    mid_x = node.x0 + node.width/2
    mid_y = node.y0 + node.height/2

    if body[0] <= mid_x:
        # left
        if body[1] <= mid_y:
            # top
            return 'top_l'
        else:
            # bot
            return 'bot_l'
    else:
        # right
        if body[1] <= mid_y:
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


def insert(tree: QuadNode, body: np.ndarray):

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

    def __init__(self, height: float, width: float):
        self.root = ExternalNode(.0, .0, width, height, None)

    def insert(self, body: np.ndarray):
        self.root = insert(self.root, body)

    def insert_list(self, bodies: list[type[np.ndarray]]):
        for b in bodies:
            self.insert(b)


def plot_quadtree(tree: QuadNode, ax: plt.Axes):

    if tree.type is NodeType.EXTERNAL:
        if tree.body is not None:
            body = tree.body
            ax.plot(body[0], body[1], 'ro', zorder=3)
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


tree = QuadTree(100.0, 100.0)
bodies = random_bodies(100)
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
