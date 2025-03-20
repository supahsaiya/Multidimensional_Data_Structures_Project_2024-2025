class RTreeNode:
    def __init__(self, is_leaf=True):
        """
        Initialize an R-Tree node.
        :param is_leaf: Whether this node is a leaf.
        """
        self.is_leaf = is_leaf
        self.entries = []  # Stores either child nodes or data points
        self.bounding_box = None  # Bounding rectangle

class RTree:
    def __init__(self, max_entries=4):
        """
        Initialize the R-Tree.
        :param max_entries: Maximum number of entries in a node before splitting.
        """
        self.max_entries = max_entries
        self.root = RTreeNode()

    def insert(self, point):
        """
        Insert a point into the R-Tree.
        :param point: A point represented as (x, y).
        """
        leaf = self.choose_leaf(self.root, point)
        leaf.entries.append(point)
        if len(leaf.entries) > self.max_entries:
            self.split_node(leaf)

    def choose_leaf(self, node, point):
        """
        Choose the appropriate leaf node for insertion.
        :param node: Current node.
        :param point: Point to insert.
        :return: The chosen leaf node.
        """
        if node.is_leaf:
            return node

        # Choose the child node with the smallest enlargement of bounding box
        best_child = None
        min_enlargement = float('inf')

        for child in node.entries:
            enlargement = self.calculate_enlargement(child.bounding_box, point)
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_child = child

        return self.choose_leaf(best_child, point)

    def calculate_enlargement(self, bounding_box, point):
        """
        Calculate how much a bounding box needs to enlarge to include a point.
        :param bounding_box: Tuple (xmin, ymin, xmax, ymax).
        :param point: Tuple (x, y).
        :return: Enlargement area.
        """
        xmin, ymin, xmax, ymax = bounding_box
        px, py = point
        new_xmin = min(xmin, px)
        new_ymin = min(ymin, py)
        new_xmax = max(xmax, px)
        new_ymax = max(ymax, py)

        old_area = (xmax - xmin) * (ymax - ymin)
        new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)

        return new_area - old_area

    def split_node(self, node):
        """
        Split a node into two when it overflows.
        :param node: Node to split.
        """
        # Placeholder: Implement splitting logic here
        pass

    def range_query(self, node, query_box):
        """
        Perform a range query on the R-Tree.
        :param node: Current node.
        :param query_box: Tuple (xmin, ymin, xmax, ymax).
        :return: List of points within the query box.
        """
        if node.is_leaf:
            # Return points within the query box
            return [
                point
                for point in node.entries
                if self.is_point_in_box(point, query_box)
            ]

        # Check child nodes whose bounding boxes intersect the query box
        results = []
        for child in node.entries:
            if self.is_box_intersecting(child.bounding_box, query_box):
                results += self.range_query(child, query_box)

        return results

    def is_point_in_box(self, point, box):
        """
        Check if a point is inside a bounding box.
        :param point: Tuple (x, y).
        :param box: Tuple (xmin, ymin, xmax, ymax).
        :return: True if the point is inside the box, False otherwise.
        """
        px, py = point
        xmin, ymin, xmax, ymax = box
        return xmin <= px <= xmax and ymin <= py <= ymax

    def is_box_intersecting(self, box1, box2):
        """
        Check if two bounding boxes intersect.
        :param box1: Tuple (xmin, ymin, xmax, ymax).
        :param box2: Tuple (xmin, ymin, xmax, ymax).
        :return: True if the boxes intersect, False otherwise.
        """
        return not (
            box1[2] < box2[0] or box1[0] > box2[2] or
            box1[3] < box2[1] or box1[1] > box2[3]
        )