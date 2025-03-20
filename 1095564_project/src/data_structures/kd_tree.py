class KDTreeNode:
    def __init__(self, point, left=None, right=None):
        self.point = point  # The k-dimensional point
        self.left = left  # Left subtree
        self.right = right  # Right subtree

class KDTree:
    def __init__(self, points=None, depth=0):
        self.node = None
        if points:
            self.node = self.build_tree(points, depth)

    def build_tree(self, points, depth):
        if not points:
            return None

        k = len(points[0])  # Dimensionality
        axis = depth % k
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2

        return KDTreeNode(
            point=points[median],
            left=self.build_tree(points[:median], depth + 1),
            right=self.build_tree(points[median + 1:], depth + 1)
        )

    def insert(self, root, point, depth=0):
        if root is None:
            return KDTreeNode(point)

        # Calculate the current dimension
        k = len(point)
        axis = depth % k

        # Recursively insert to the left or right subtree
        if point[axis] < root.point[axis]:
            root.left = self.insert(root.left, point, depth + 1)
        else:
            root.right = self.insert(root.right, point, depth + 1)

        return root

    def insert_point(self, point):
        # Public method to insert a point into the tree
        self.node = self.insert(self.node, point)

    def range_search(self, root, query_range, depth=0):
        if root is None:
            return []

        point = root.point
        k = len(query_range)
        axis = depth % k

        # Check if the current point is in range
        in_range = all(
            query_range[dim][0] <= point[dim] <= query_range[dim][1]
            for dim in range(k)
        )
        results = [point] if in_range else []

        # Traverse the left and right subtrees
        if query_range[axis][0] <= point[axis]:
            results += self.range_search(root.left, query_range, depth + 1)
        if query_range[axis][1] >= point[axis]:
            results += self.range_search(root.right, query_range, depth + 1)

        return results

    def search_range(self, query_range):
        return self.range_search(self.node, query_range)
