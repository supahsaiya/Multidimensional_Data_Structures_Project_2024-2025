class QuadTreeNode:
    def __init__(self, boundary, capacity):
        """
        Initialize a QuadTree node.
        :param boundary: Tuple (x_min, x_max, y_min, y_max) defining the region.
        :param capacity: Maximum number of points a node can hold before splitting.
        """
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.nw = self.ne = self.sw = self.se = None  # Child nodes

    def contains(self, point):
        """
        Check if the point is within the boundary of this node.
        :param point: Tuple (x, y).
        """
        x, y = point
        x_min, x_max, y_min, y_max = self.boundary
        return x_min <= x <= x_max and y_min <= y <= y_max

    def subdivide(self):
        """
        Subdivide the node into four child nodes.
        """
        x_min, x_max, y_min, y_max = self.boundary
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        self.nw = QuadTreeNode((x_min, x_mid, y_mid, y_max), self.capacity)
        self.ne = QuadTreeNode((x_mid, x_max, y_mid, y_max), self.capacity)
        self.sw = QuadTreeNode((x_min, x_mid, y_min, y_mid), self.capacity)
        self.se = QuadTreeNode((x_mid, x_max, y_min, y_mid), self.capacity)

        self.divided = True

class QuadTree:
    def __init__(self, boundary, capacity):
        """
        Initialize a QuadTree.
        :param boundary: Tuple (x_min, x_max, y_min, y_max) defining the root region.
        :param capacity: Maximum number of points a node can hold before splitting.
        """
        self.root = QuadTreeNode(boundary, capacity)

    def insert(self, node, point):
        """
        Insert a point into the QuadTree.
        :param node: Current node in the QuadTree.
        :param point: Tuple (x, y).
        """
        if not node.contains(point):
            return False

        if len(node.points) < node.capacity:
            node.points.append(point)
            return True

        if not node.divided:
            node.subdivide()

        # Recursively insert into appropriate quadrant
        return (
            self.insert(node.nw, point) or
            self.insert(node.ne, point) or
            self.insert(node.sw, point) or
            self.insert(node.se, point)
        )

    def range_search(self, node, range_boundary):
        """
        Perform a range query on the QuadTree.
        :param node: Current node in the QuadTree.
        :param range_boundary: Tuple (x_min, x_max, y_min, y_max).
        :return: List of points within the range.
        """
        if node is None:
            return []

        x_min, x_max, y_min, y_max = range_boundary
        node_x_min, node_x_max, node_y_min, node_y_max = node.boundary

        # Check if node's boundary intersects the range
        if node_x_max < x_min or node_x_min > x_max or node_y_max < y_min or node_y_min > y_max:
            return []

        # Collect points within the range
        results = [
            point for point in node.points
            if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
        ]

        # Search in child nodes
        if node.divided:
            results += self.range_search(node.nw, range_boundary)
            results += self.range_search(node.ne, range_boundary)
            results += self.range_search(node.sw, range_boundary)
            results += self.range_search(node.se, range_boundary)

        return results

    def insert_point(self, point):
        """
        Public method to insert a point into the QuadTree.
        """
        self.insert(self.root, point)

    def search_range(self, range_boundary):
        """
        Public method to perform a range search.
        """
        return self.range_search(self.root, range_boundary)
