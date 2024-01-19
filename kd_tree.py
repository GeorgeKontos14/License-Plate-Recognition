import numpy as np
import heapq

class HeapEntry:
    def __init__(self, dist: float, point: np.ndarray, img: np.ndarray, label: int) -> None:
        """ Initialiser of HeapEntry class

            Parameters:
            dist (float): distance between target point and given point
            point (np.ndarray): feature vector 
            label (int): label of speecific data object
            img (np.ndarray): character represented by an multi dimensional array

            Returns:
            None
        """
        self.dist: float= dist
        self.point: np.ndarray = point
        self.img: np.ndarray = img
        self.label: int = label

    def __repr__(self) -> str:
        """ Returns object information as string

            Parameters:
            None

            Returns:
            str: object information as string
        """
        return f'({self.dist}, {self.point}, {self.label})'
    
    def __lt__(self, other) -> bool:
        """ Compares distances of self.point and some other point

            Parameters:
            other (HeapEntry): object to compare with

            Returns:
            bool: whether the self.distance is less than the given point
        """
        return self.dist < other.dist

class KDNode:
    def __init__(self, point: np.ndarray, img: np.ndarray, label: int, d: int, dim: int) -> None:
        """ Initialiser of Node class

            Parameters:
            point (np.ndarray): point in D-dimensional space that the kd-node holds
            img (np.ndarray): character represented by an multi dimensional array
            d (int): dimensions of the tree
            dim (int): split dimension

            Returns:
            None
        """
        self.point: np.ndarray = point
        self.d: int = d
        self.dim: int = dim
        self.img: np.ndarray = img
        self.label: int = label

        self.left: KDNode = None
        self.right: KDNode = None

    def isPointToLeft(self, point: np.ndarray) -> bool:
        """ Returns whether the give point has a smaller 
            or equal value to the node's dimension


            Parameters:
            point (np.ndarray): point to compare

            Returns:
            bool: Whether the give point has a smaller 
                  or equal value to the node's dimension  
        """
        return self.point[self.dim] >= point[self.dim]
    

class KDTree:
    def __init__(self) -> None:
        """ Initialiser of class KDTree

            Parameters:
            root (KDNode)

            Returns:
            None
        """
        self.root = None
    
    def insert_point(self, point: np.ndarray, img: np.ndarray, label: int) -> None:
        """ Inserts given point to KD-tree

            Parameters:
            node (KDNode)
            point (np.ndarray)
            label (int)

            Returns:
            None
        """
        if self.root is None:
            self.root = KDNode(point, img, label, point.shape[0], 0)
            return
        KDTree.insert_point_helper(self.root, point, img, label)

    def get_k_nearest_points(self, point: np.ndarray, k: int) -> list:
        """ Gets the k nearest neighbours of a given point

            Parameters:
            point (np.ndarray): Target point
            k (int): Number of neighbours to look for

            Returns:
            list: The k nearest neighbours of a given point
        """
        k_nearest_points: list = []

        KDTree.get_k_nearest_points_helper(self.root, point, k, k_nearest_points)

        return k_nearest_points

    def insert_point_helper(node: KDNode, point: np.ndarray, img: np.ndarray, label: int) -> None:
        """ Inserts given point to KD-tree

            Parameters:
            node (KDNode)
            point (np.ndarray)

            Returns:
            None
        """
        if node is None:
            return
        
        next_dim: int = node.dim + 1 if node.dim + 1 < node.d else 0

        if node.isPointToLeft(point):
            if node.left is None:
                node.left = KDNode(point, img, label, point.shape[0], next_dim)
            else:
                KDTree.insert_point_helper(node.left, point, img, label)
        else:
            if node.right is None:
                node.right = KDNode(point, img, label, point.shape[0], next_dim)
            else:
                KDTree.insert_point_helper(node.right, point, img, label)

    def get_k_nearest_points_helper(
            node: KDNode, 
            point: np.ndarray, 
            k: int,
            k_nearest_points: list
        ) -> None:
        """ Gets the k nearest neighbours of a given point

            Parameters:
            node (KDNode): Current node in the tree
            point (np.ndarray): Target point
            k (int): Number of neighbours to look for
            k_nearest_points (list): A heap containing the k nearest neighbours 
                                     at a given point of tranversal 

            Returns:
            None
        """

        if node is None:
            return 
        
        opposite_branch: KDNode = None

        if node.isPointToLeft(point):
            KDTree.get_k_nearest_points_helper(node.left, point, k, k_nearest_points)
            opposite_branch = node.right
        else:
            KDTree.get_k_nearest_points_helper(node.right, point, k, k_nearest_points)
            opposite_branch = node.left

        curr_dist: float = euclidean_distance(node.point, point)

        if len(k_nearest_points) < k or curr_dist < k_nearest_points[0].dist:
            heapq.heappush(k_nearest_points, HeapEntry(curr_dist, node.point, node.img, node.label))
            
            if len(k_nearest_points) > k:
                heapq.heappop(k_nearest_points)

        if opposite_branch is not None:
            if np.sqrt(np.square(node.point[node.dim] - point[node.dim])) < k_nearest_points[0].dist:
                KDTree.get_k_nearest_points_helper(opposite_branch, point, k, k_nearest_points)
        

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Calculates the euclidean distance between two points

        Parameters:
        p1 (np.ndarray) 
        p2 (np.ndarray)

        Returns:
        float: The distance between the two given points
    """
    return np.sqrt(np.sum(np.square(p1 - p2)))