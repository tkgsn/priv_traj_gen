import numpy as np
import pickle

class Grid():
    # A Grid instance has a bidirectional mapping between a state and a lat/lon pair
    # Each cell size is variable

    @staticmethod
    def make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins):
        x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
        y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)

        # compute the bin size
        x_bin_size = x_axis[1] - x_axis[0]
        y_bin_size = y_axis[1] - y_axis[0]

        # insert the first element (-infty, x_axis[0]) and the last element (x_axis[-1], +infty)
        x_axis = np.insert(x_axis, 0, x_axis[0]-x_bin_size)
        x_axis = np.append(x_axis, x_axis[-1]+x_bin_size)

        # insert the first element (-infty, y_axis[0]) and the last element (y_axis[-1], +infty)
        y_axis = np.insert(y_axis, 0, y_axis[0]-y_bin_size)
        y_axis = np.append(y_axis, y_axis[-1]+y_bin_size)

        ranges = []
        for i in range(len(x_axis)-1):
            for j in range(len(y_axis)-1):
                ranges.append([(x_axis[i], x_axis[i+1]), (y_axis[j], y_axis[j+1])])
        return ranges

    @staticmethod
    def make_ranges_from_privtrace_info(info_path):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        x_ranges, y_ranges = info
        ranges = []
        for x_ranges_level2, y_ranges_level2 in zip(x_ranges, y_ranges):
            if len(x_ranges_level2) == 2:
                ranges.append([(x_ranges_level2[0], x_ranges_level2[1]), (y_ranges_level2[0], y_ranges_level2[1])])
            else:
                ranges += [[(x_ranges_level2[i], x_ranges_level2[i+1]), (y_ranges_level2[j], y_ranges_level2[j+1])] for i in range(len(x_ranges_level2)-1) for j in range(len(y_ranges_level2)-1)]
        return ranges

    def __init__(self, ranges):
        self.grids = self.make_grid_from_ranges(ranges)
        assert not self.check_grid_overlap(), "Grids overlap"
        self.vocab_size = len(self.grids)
        self.max_distance = self.compute_max_distance()

    def compute_max_distance(self):
        max_distance = 0
        for i, (x_range, y_range) in self.grids.items():
            for j, (x_range2, y_range2) in self.grids.items():
                if i != j:
                    max_distance = max(max_distance, np.sqrt((x_range[0]-x_range2[0])**2 + (y_range[0]-y_range2[0])**2))
        return max_distance

    def save_gps(self, gps_path):

        with open(gps_path, "w") as f:
            for state in self.grids:
                lon_center, lat_center = self.state_to_center_latlon(state)
                f.write(f"{lat_center},{lon_center}\n")


    def make_grid_from_ranges(self, ranges):
        grids = {}
        for i, (x_range, y_range) in enumerate(ranges):
            grids[i] = [x_range, y_range]
        return grids

    def state_to_center_latlon(self, state):
        x_range, y_range = self.grids[state]
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        return y_center, x_center

    def state_to_random_latlon_in_the_cell(self, state):
        x_range, y_range = self.grids[state]
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        return y, x

    # check if the grids have overlap
    def check_grid_overlap(self):
        for i, (x_range, y_range) in self.grids.items():
            for j, (x_range2, y_range2) in self.grids.items():
                if i != j and x_range[0] < x_range2[0] < x_range[1] and y_range[0] < y_range2[0] < y_range[1]:
                    print(x_range, x_range2, y_range, y_range2)
                    return True
        return False

    # convert latlon to state by bisect search
    def latlon_to_state(self, lat, lon):
        for state, (x_range, y_range) in self.grids.items():
            if x_range[0] <= lon < x_range[1] and y_range[0] <= lat < y_range[1]:
                return state
        return None
    
    def register_count(self, counts):
        self.counts = counts

class Node():
    def __init__(self, depth, state_list=None, children=None):
        assert state_list is not None or children is not None, "state_list or children must be given"
        assert state_list is None or children is None, "state_list and children cannot be given at the same time"
        # state_list is a list of states
        self.state_list = state_list
        # children is a list of 4 nodes
        self.children = children
        self.unvisited = True
        self.set_depth(depth)

    @staticmethod
    def make_root_node(state_list):
        root_node = Node(state_list=state_list, depth=0)
        return root_node

    def set_visited(self):
        self.unvisited = False

    def set_count(self, count):
        self.count = count

    def set_noisy_count(self, noisy_count):
        self.noisy_count = noisy_count

    def set_depth(self, depth):
        self.depth = depth

    def get_depth(self):
        return self.depth


class QuadTree(Grid):
    # recursively divide the space into 4 quadrants
    # -----------
    # | q1 | q2 |
    # -----------
    # | q3 | q4 |
    # -----------
    # each node is a set of states
    # quat_tree: a graph where each node is a set of states
    # the nodes of the finest level are the states defined by the grid
    # that is, n_bins +2 must be a power of 2

    def __init__(self, ranges):
        super().__init__(ranges)
        # check if self.vocab_size is a power of 2
        assert self.vocab_size & (self.vocab_size - 1) == 0, "self.vocab_size must be a power of 2"
        self.root_node = Node.make_root_node(list(range(self.vocab_size)))

    @staticmethod
    def divide(node):
        assert node.state_list is not None, "input must be a leaf"
        assert len(node.state_list) % 4 == 0, f"node.state_list: {node.state_list} must be divided into 4"
        # check if len(node.state_list) is a power of 2
        assert len(node.state_list) & (len(node.state_list) - 1) == 0, "len(node.state_list) must be a power of 2"
        length = int(np.sqrt(len(node.state_list)))
        depth = node.get_depth()

        # q1 is the locations on the upper left
        # q2 is the locations on the upper right
        # q3 is the locations on the lower left
        # q4 is the locations on the lower right
        # locations are located in the order of the state number from the upper left to the lower right
        q1_indice = [range(i,i+int(length/2)) for i in range(0,int(length*length/2),length)]
        q1_indice = [item for sublist in q1_indice for item in sublist]
        q2_indice = [range(i+int(length/2),i+length) for i in range(0,int(length*length/2),length)]
        q2_indice = [item for sublist in q2_indice for item in sublist]
        q3_indice = [range(i,i+int(length/2)) for i in range(int(length*length/2),length*length,length)]
        q3_indice = [item for sublist in q3_indice for item in sublist]
        q4_indice = [range(i+int(length/2),i+length) for i in range(int(length*length/2),length*length,length)]
        q4_indice = [item for sublist in q4_indice for item in sublist]
        q1 = Node(depth+1, state_list=[node.state_list[i] for i in q1_indice])
        q2 = Node(depth+1, state_list=[node.state_list[i] for i in q2_indice])
        q3 = Node(depth+1, state_list=[node.state_list[i] for i in q3_indice])
        q4 = Node(depth+1, state_list=[node.state_list[i] for i in q4_indice])
        node.children = [q1, q2, q3, q4]
        node.state_list = None
        node.unvisited = False

    
    # get leafs from the root node by recursion
    def get_leafs(self):
        return list(self._get_leafs(self.root_node))
    
    def get_unvisited_leafs(self):
        return list(self._get_unvisited_leafs(self.root_node))
    
    def _get_leafs(self, node):
        if node.children is None:
            yield node
        else:
            for child in node.children:
                yield from self._get_leafs(child)

    def _get_unvisited_leafs(self, node):
        if node.children is None and node.unvisited:
            yield node
        else:
            if node.children is not None:
                for child in node.children:
                    yield from self._get_unvisited_leafs(child)
                
    def count_of_node(self, node):
        assert node.state_list is not None, "input must be a leaf"
        return sum([self.counts[state] for state in node.state_list])
    

def laplace_noise(Lambda, seed=7): # using inverse transform sampling
    # for numbers between -N and N
    N = Lambda*10
    x = np.arange(-N,N+1,N/20000)

    # pdf P
    P = 1.0 / (2*Lambda) * np.exp(-np.abs(x) / Lambda)
    P = P / np.sum(P)
    
    # cdf C
    C = P.copy()
    for i in np.arange(1, P.shape[0]):
        C[i] = C[i-1] + P[i]
    
    # get sample from laplace distribution wiht uniform random number
    u = np.random.rand()
    sample = x[np.argmin(np.abs(C-u))]
    
    return sample

# priv_tree
def priv_tree(quad_tree, lam=2, theta=1000, delta=10, seed=0):
    # simple tree parameters
    #x, y = data['longitude'].values, data['latitude'].values
    #lam = laplace noise parameter
    #theta = 50 #min count per domain
    #h = 10 # max tree depth
    np.random.seed(seed)
    
    unvisited_leafs = quad_tree.get_unvisited_leafs()

    # create subdomains where necessary
    while unvisited_leafs != []: # while unvisited_domains is not empty
        for unvisited_leaf in unvisited_leafs:
            count = quad_tree.count_of_node(unvisited_leaf)
            tree_depth = unvisited_leaf.get_depth()
            b = count - (delta*tree_depth)
            b = max(b, (theta - delta))
            noisy_b = b + laplace_noise(lam)

            if (noisy_b > theta) and len(unvisited_leaf.state_list) != 1: #split if condition is met
                QuadTree.divide(unvisited_leaf)
            else:
                # remove domain that was just visited
                unvisited_leaf.set_visited()
                # record count and noisy count
                unvisited_leaf.set_count(count)
                unvisited_leaf.set_noisy_count(noisy_b)
        unvisited_leafs = quad_tree.get_unvisited_leafs()