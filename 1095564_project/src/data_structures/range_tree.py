class RangeTreeNode:
    def __init__(self, point):
        """
        Initialize a Range Tree node.
        :param point: Tuple (x, y).
        """
        self.point = point
        self.left = None
        self.right = None
        self.secondary_tree = None  # Secondary BST for the second dimension

class RangeTree:
    def __init__(self, points):
        """
        Build the range tree from a list of points.
        :param points: List of tuples [(x1, y1), (x2, y2), ...].
        """
        self.root = self.build_tree(points, depth=0)

    def build_tree(self, points, depth):
        """
        Recursively build the range tree.
        :param points: List of points.
        :param depth: Current depth in the tree.
        """
        if not points:
            return None

        # Sort points by the current dimension
        axis = depth % 2  # 0 for x, 1 for y
        points.sort(key=lambda p: p[axis])
        median_idx = len(points) // 2

        # Create the node and recursively build subtrees
        node = RangeTreeNode(points[median_idx])
        node.left = self.build_tree(points[:median_idx], depth + 1)
        node.right = self.build_tree(points[median_idx + 1:], depth + 1)

        # Build the secondary tree for the y-dimension at the root
        if depth == 0:
            sorted_by_y = sorted(points, key=lambda p: p[1])
            node.secondary_tree = self.build_tree(sorted_by_y, depth=1)

        return node

    def range_query(self, node, query_range, depth=0):
        """
        Perform a range query on the range tree.
        :param node: Current node in the tree.
        :param query_range: List of tuples [(min1, max1), (min2, max2)].
        :param depth: Current depth in the tree.
        :return: List of points within the range.
        """
        if node is None:
            return []

        axis = depth % 2
        min_val, max_val = query_range[axis]
        x, y = node.point

        results = []

        # Check if the current point is within the range
        if all(
            query_range[dim][0] <= node.point[dim] <= query_range[dim][1]
            for dim in range(2)
        ):
            results.append(node.point)

        # Traverse the left and right subtrees
        if min_val <= node.point[axis]:
            results += self.range_query(node.left, query_range, depth + 1)
        if max_val >= node.point[axis]:
            results += self.range_query(node.right, query_range, depth + 1)

        return results

    def search_range(self, query_range):
        """
        Public method to perform a range query.
        :param query_range: List of tuples [(min1, max1), (min2, max2)].
        """
        return self.range_query(self.root, query_range)
