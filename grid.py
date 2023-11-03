import numpy as np
import pickle
import torch

class Grid():
    # A Grid instance has a bidirectional mapping between a state and a lat/lon pair
    # Each cell size is variable

    @staticmethod
    def make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins):
        x_axis = np.linspace(lon_range[0]-1e-5, lon_range[1]+1e-5, n_bins+3)
        y_axis = np.linspace(lat_range[0]-1e-5, lat_range[1]+1e-5, n_bins+3)

        # compute the bin size
        x_bin_size = x_axis[1] - x_axis[0]
        y_bin_size = y_axis[1] - y_axis[0]

        ranges = []
        for i in range(len(x_axis)-1):
            for j in range(len(y_axis)-1):
                ranges.append([(x_axis[i], x_axis[i+1]), (y_axis[j], y_axis[j+1])])
        return ranges

    # @staticmethod
    # def make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins):
    #     x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
    #     y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)

    #     # compute the bin size
    #     x_bin_size = x_axis[1] - x_axis[0]
    #     y_bin_size = y_axis[1] - y_axis[0]

    #     # insert the first element (-infty, x_axis[0]) and the last element (x_axis[-1], +infty)
    #     x_axis = np.insert(x_axis, 0, x_axis[0]-x_bin_size)
    #     x_axis = np.append(x_axis, x_axis[-1]+x_bin_size)

    #     # insert the first element (-infty, y_axis[0]) and the last element (y_axis[-1], +infty)
    #     y_axis = np.insert(y_axis, 0, y_axis[0]-y_bin_size)
    #     y_axis = np.append(y_axis, y_axis[-1]+y_bin_size)

    #     ranges = []
    #     for i in range(len(x_axis)-1):
    #         for j in range(len(y_axis)-1):
    #             ranges.append([(x_axis[i], x_axis[i+1]), (y_axis[j], y_axis[j+1])])
    #     return ranges

    # @staticmethod
    # def make_ranges_from_privtrace_info(info_path):
    #     with open(info_path, "rb") as f:
    #         info = pickle.load(f)
    #     x_ranges, y_ranges = info
    #     ranges = []
    #     for x_ranges_level2, y_ranges_level2 in zip(x_ranges, y_ranges):
    #         if len(x_ranges_level2) == 2:
    #             ranges.append([(x_ranges_level2[0], x_ranges_level2[1]), (y_ranges_level2[0], y_ranges_level2[1])])
    #         else:
    #             ranges += [[(x_ranges_level2[i], x_ranges_level2[i+1]), (y_ranges_level2[j], y_ranges_level2[j+1])] for i in range(len(x_ranges_level2)-1) for j in range(len(y_ranges_level2)-1)]
    #     return ranges

    def __init__(self, ranges):
        self.grids = self.make_grid_from_ranges(ranges)
        assert not self.check_grid_overlap(), "Grids overlap"
        self.vocab_size = len(self.grids)
        self.max_distance = self.compute_max_distance()
        # if the number of grid is a square number, register n_bins
        if np.sqrt(self.vocab_size) % 1 == 0:
            self.n_bins = int(np.sqrt(self.vocab_size))-2
        self.lat_range, self.lon_range = self.compute_latlon_range()

    def compute_latlon_range(self):
        lat_range = [np.inf, -np.inf]
        lon_range = [np.inf, -np.inf]
        for (x_range, y_range) in self.grids.values():
            lat_range[0] = min(lat_range[0], y_range[0])
            lat_range[1] = max(lat_range[1], y_range[1])
            lon_range[0] = min(lon_range[0], x_range[0])
            lon_range[1] = max(lon_range[1], x_range[1])
        return lat_range, lon_range

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

    def is_in_range(self, lat, lon):
        return self.lat_range[0]-1e-5 <= lat < self.lat_range[1]+1e-5 and self.lon_range[0]-1e-5 <= lon < self.lon_range[1]+1e-5

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

    def get_count(self):
        if not hasattr(self, "count"):
            self.count = sum([child.get_count() for child in self.children])
            return self.count
        else:
            return self.count

    def set_noisy_count(self, noisy_count):
        self.noisy_count = noisy_count

    def set_depth(self, depth):
        self.depth = depth

    def get_depth(self):
        return self.depth
    
    def is_leaf(self):
        return self.children is None


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
        # depth is the log of the vocab_size with the base 4
        self.max_depth = int(np.log2(self.vocab_size)/2)
        self.state_to_path_cache = {}
        self.state_to_node_path_cache = {}
        self.get_nodes_cache = {}
        self.is_complete = False

    @staticmethod
    def divide(node):
        # if the node is the leaf at the finest level, return False
        if len(node.state_list) == 1:
            return False
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
        node.unvisited = False
        q1.parent = node
        q2.parent = node
        q3.parent = node
        q4.parent = node
        return True
    
    def make_self_complete(self):
        # if all nodes are divided, condition becomes False
        condition = True
        while condition:
            leafs = self.get_leafs()
            for leaf in leafs:
                condition = condition and QuadTree.divide(leaf)
        self.register_id()
        self.node_id_to_hidden_id = self._make_hidden_ids()
        self.hidden_id_to_node_id = {hidden_id:node_id for node_id, hidden_id in enumerate(self.node_id_to_hidden_id)}
        self.location_id_to_node_id = self._make_location_id_to_node_id()
        self.is_complete = True

    def _make_location_id_to_node_id(self):
        location_id_to_node_id = {}
        leafs = self.get_leafs()
        for node in self.get_leafs():
            for state in node.state_list:
                location_id_to_node_id[state] = node.id
        location_id_to_node_id[len(leafs)] = len(self.get_all_nodes())-1
        location_id_to_node_id[len(leafs)+1] = len(self.get_all_nodes())
        return location_id_to_node_id

    # hidden id is the id labeld by the order from the upper left to the lower right
    def _make_hidden_ids(self):
        nodes = self.get_all_nodes()
        self.set_coordinate()

        id_to_hidden_id = {}
        for depth in range(1,self.max_depth+1):
            nodes = self.get_nodes(depth)
            base_number = nodes[0].id-1
            for node in nodes:
                id_to_hidden_id[node.id] = node.oned_coordinate+base_number
        
        # sort by key
        id_to_hidden_id = dict(sorted(id_to_hidden_id.items(), key=lambda x:x[0]))
        
        # the root node does not correspond to any location
        return [0] + list(id_to_hidden_id.values())

    def set_coordinate(self):
        nodes = self.get_all_nodes()
        # set place_id (0, 1, 2, 3) for each node
        # place_id is the index of the node in the children list of the parent node
        for node in nodes:
            if hasattr(node, "parent"):
                node._place_id = node.parent.children.index(node)

        # if the coordinate of the parent is (x,y), 
        # the coordinate of the children are (2x, 2y), (2x+1, 2y), (2x, 2y+1), (2x+1, 2y+1)
        for node in self.get_all_nodes():
            if hasattr(node, "parent") is False:
                node.coordinate = (0, 0)
            else:
                parent_coordinate = node.parent.coordinate
                place_id = node._place_id
                node.coordinate = (parent_coordinate[0]*2 + place_id % 2, parent_coordinate[1]*2 + place_id // 2)
                node.oned_coordinate = node.coordinate[0] + node.coordinate[1] * 2**node.depth

    def reset_count(self):
        # remove count from all nodes
        for i in range(self.max_depth+1):
            nodes = self.get_nodes(i)
            for node in nodes:
                if hasattr(node, "count"):
                    delattr(node, "count")


    def make_quad_distribution(self, counts):
        # assert self.is_complete
        # to batched shape
        # if len(counts.shape) == 1:
        #     counts = counts.rehsape(1, -1)
        # self._register_count_to_complete_graph(counts)
        # batch_size = counts.shape[0]

        # nodes_except_leafs = self.get_all_nodes()
        # nodes_except_leafs = [node for node in nodes_except_leafs if node.is_leaf() is False]
        # quad_distribution = np.zeros(batch_size, len(nodes_except_leafs), 4)

        # for node in nodes_except_leafs:
        #     for i, child in enumerate(node.children):
        #         quad_distribution[:,node.id,i] = child.get_count()

        # return quad_distribution
        assert self.is_complete
        # to batched shape
        if len(counts.shape) == 1:
            counts = counts.rehsape(1, -1)
        batch_size = counts.shape[0]
        n_leafs = counts.shape[1]

        # depth is log of n_leafs with the base 4
        depth = int(np.log2(n_leafs)/2)
        n_nodes_except_leafs = sum([4**depth_ for depth_ in range(depth)])
        quad_distribution = torch.zeros((batch_size, n_nodes_except_leafs, 4)).to(counts.device)
        for i in range(n_leafs):
            node_path = self.state_to_node_id_path(i)[:-1]
            place_path = self.state_to_path(i)
            quad_distribution[:, node_path, place_path] += counts[:, i].view(-1, 1)
        quad_distribution = torch.nn.functional.normalize(quad_distribution, p=1, dim=-1)
        return quad_distribution


    def _register_count_to_complete_graph(self, counts):
        # counts: batch_size, n_locations
        assert self.is_complete
        leafs = self.get_leafs()
        for leaf in leafs:
            assert len(leaf.state_list) == 1, "leaf.state_list must be a singleton"
            state = leaf.state_list[0]
            leaf.set_count(counts[:,state])

    def register_id(self):
        # set id from 0 to len(nodes)-1
        id = 0
        nodes = self.get_all_nodes()
        for node in nodes:
            node.id = id
            id += 1

    def get_node_by_id(self, id):
        return self.get_all_nodes()[id]

    def get_all_nodes(self):
        if hasattr(self, "all_nodes"):
            return self.all_nodes

        nodes = []
        for i in range(self.max_depth+1):
            nodes += self.get_nodes(i)
        
        self.all_nodes = nodes
        return nodes
        # return list(self._get_all_nodes(self.root_node))
    
    # def _get_all_nodes(self, node):
        # yield node
        # if node.children is not None:
        #     for child in node.children:
        #         yield from self._get_all_nodes(child)
        
    # get nodes at the depth
    def get_nodes(self, depth):
        if self.is_complete and (depth in self.get_nodes_cache):
            return self.get_nodes_cache[depth]
        else:
            nodes = list(self._get_nodes(self.root_node, depth))
            self.get_nodes_cache[depth] = nodes
            assert len(nodes) == 4**depth, f"len(nodes): {len(nodes)} must be 4**(depth-1): {4**depth}"
            return nodes
    
    def _get_nodes(self, node, depth):
        if node.depth == depth:
            yield node
        else:
            if node.children is not None:
                for child in node.children:
                    yield from self._get_nodes(child, depth)
    
    # get leafs from the root node by recursion
    def get_leafs(self):
        if hasattr(self, "leafs"):
            if all([leaf.children is None for leaf in self.leafs]):
                return self.leafs
        leafs = list(self._get_leafs(self.root_node))
        self.leafs = leafs
        return self.leafs
    
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
        if node.is_leaf():
            return sum([self.counts[state] for state in node.state_list])
        else:
            return sum([self.count_of_node(child) for child in node.children])
        
    def get_leaf_ids_in_tree(self, tree):
        leafs = self.get_leafs()
        leaf_id_in_tree = []
        for leaf in leafs:
            depth = leaf.get_depth()
            nodes = tree.get_nodes(depth)
            condition = False
            for node in nodes:
                if node.state_list == leaf.state_list:
                    leaf_id_in_tree.append(node.id)
                    condition = True
                    leaf.id = node.id
                    break
            assert condition, "leaf_id_in_tree must be found"
                # if all nodes are not the same, raise
        return leaf_id_in_tree
    
    def state_to_path(self, state):
        if state in self.state_to_path_cache:
            return self.state_to_path_cache[state]
        else:
            node = self.get_node_by_state(state)
            if node is None:
                self.state_to_path_cache[state] = [4]*self.max_depth
                return [4]*self.max_depth
            path = []
            while hasattr(node, "parent"):
                path.append(node._place_id)
                node = node.parent
            self.state_to_path_cache[state] = path[::-1]
            return path[::-1]
    
    def state_to_node_id_path(self, state):
        if state in self.state_to_node_path_cache:
            node_path = self.state_to_node_path_cache[state]
        else:
            node_path = self.state_to_node_path(state)
        return [node.id for node in node_path]

    def state_to_node_path(self, state):
        if state in self.state_to_node_path_cache:
            return self.state_to_node_path_cache[state]
        else:
            node = self.get_node_by_state(state)
            if node is None:
                self.state_to_node_path_cache[state] = [None]*self.max_depth
                return [None]*self.max_depth
            path = []
            while hasattr(node, "parent"):
                path.append(node)
                node = node.parent
            path.append(self.root_node)
            self.state_to_node_path_cache[state] = path[::-1]
            return path[::-1]

    
    def get_node_by_state(self, state):
        for node in self.get_leafs():
            assert len(node.state_list) == 1, "node.state_list must be a singleton"
            if state in node.state_list:
                return node
        return None
    
    def get_location_id_in_the_depth(self, state, depth):
        node_path = self.state_to_node_path(state)
        return self.node_id_to_hidden_id[node_path[depth].id]
    

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