import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform, morphology


def euclid_dist_between_nodes(n1, n2):
    return np.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)


def get_dir_masks():
    direction_masks = []
    for dir_mask_index in range(1, 10):
        mask = np.zeros(9)
        mask[dir_mask_index - 1] = 1
        mask = mask.reshape((3, 3))
        direction_masks.append(mask)
    return direction_masks


class Network2d:
    G = None
    near_node_tol = 5
    length_tol = 1
    ends = None
    branches = None
    img_skel = None
    img_dist = None
    img_skel_erode = None
    img_dir = None

    pos = None
    img_enhanced = np.zeros(0)

    def __init__(self, img_skel, img_dist, near_node_tol=5, length_tol=1, min_nodes=3,
                 plot=False, img_enhanced=np.zeros(0)):
        self.img_skel = img_skel
        self.img_dist = img_dist
        self.near_node_tol = near_node_tol
        self.length_tol = length_tol
        self.min_nodes = min_nodes
        self.img_enhanced = img_enhanced

        self.get_ends_and_branches()
        self.create_graph_and_nodes()
        self.fill_edges()

        self.combine_near_nodes_eculid()
        self.average_dupedge_lengths()
        self.combine_near_nodes_length()
        self.remove_zero_length_edges()

        self.remove_small_subgraphs()

        self.get_pos_dict()

        if plot:
            self.show_graph(with_pos=self.pos, with_background=img_enhanced,
                            with_skel=morphology.dilation(self.img_skel))
            self.show_graph()

    def get_ends_and_branches(self):
        print('\t - Identifying locations of branches and ends')
        ends = []
        branches = []

        img_skel = np.asarray(self.img_skel, 'uint8')
        img_skel = img_skel / (np.amax(img_skel))
        dim = img_skel.shape

        for y in range(0, dim[0]):
            for x in range(0, dim[1]):

                # Get the index of skeleton point
                if img_skel[x, y] == 1:
                    xl, xr = None, None
                    yt, yb = None, None

                    ####### Nominal case ####### (not on any edges)
                    if (0 < x < dim[0]) and (0 < y < dim[1]):
                        xl, xr = 1, 2
                        yt, yb = 1, 2

                    ####### Check Corners ######
                    # Top Left
                    elif x == 0 and y == 0:
                        xl, xr = 0, 2
                        yt, yb = 0, 2

                    # Top Right
                    elif x == dim[0] and y == 0:
                        xl, xr = 1, 1
                        yt, yb = 0, 2

                    # Bottom Right
                    elif x == dim[0] and y == dim[1]:
                        xl, xr = 1, 1
                        yt, yb = 1, 1

                    # Bottom Left
                    elif x == 0 and y == dim[1]:
                        xl, xr = 0, 2
                        yt, yb = 1, 1


                    ###### Check Edges #######
                    # Left Edge
                    elif x == 0:
                        xl, xr = 0, 2
                        yt, yb = 1, 2

                    # Top Edge
                    elif y == 0:
                        xl, xr = 1, 2
                        yt, yb = 0, 2

                    # Right Edge
                    elif x == dim[0]:
                        xl, xr = 1, 1
                        yt, yb = 1, 2

                    # Bottom Edge
                    elif y == dim[1]:
                        xl, xr = 1, 2
                        yt, yb = 1, 1

                    num_neighbors = np.sum(img_skel[x - xl: x + xr, y - yt:y + yb]) - 1
                    if num_neighbors == 1:
                        ends.append((y, x))
                    elif num_neighbors >= 3:
                        branches.append((y, x))

        self.ends = set(ends)
        self.branches = set(branches)

    def create_graph_and_nodes(self):
        print('\t - Establishing graph with branch and end nodes')
        G = nx.Graph()
        self.img_skel_erode = self.img_skel.copy()

        nodeID = 0
        for b in self.branches:
            G.add_node(b)
            self.img_skel_erode[b[1], b[0]] = 0
            nodeID += 1

        for e in self.ends:
            G.add_node(e)
            self.img_skel_erode[e[1], e[0]] = 0
            nodeID += 1

        self.G = G

    def create_direction_transform(self, loc, direction_masks):
        neighbors = self.img_skel_erode[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2]
        for i in range(0, len(direction_masks)):
            j = i + 1
            if i == 4:
                j = 0
            elif i > 4:
                j = 9 - i
            self.img_dir[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2] += j * neighbors * direction_masks[i]

    def fill_edges(self):
        print('\t - Connecting nodes with weighted edges')
        direction_masks = get_dir_masks()
        self.img_dir = np.zeros(self.img_skel_erode.shape)

        for b in self.branches:
            loc = (b[1], b[0])
            origin = b
            length_tot = 0
            width_vals = []
            self.edge_walk(loc, origin, length_tot, width_vals, direction_masks)

    def edge_walk(self, loc, origin, length_tot, width_vals, direction_masks):
        loc = (int(loc[0]), int(loc[1]))
        self.img_skel_erode[loc] = 0
        self.create_direction_transform(loc, direction_masks)

        neighbors = np.where(self.img_skel_erode[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2])
        N = int(np.sum(self.img_skel_erode[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2]))
        for n in range(0, N):
            width_vals.append(self.img_dist[loc])
            self.edge_walk((neighbors[0][n] - 1 + loc[0], neighbors[1][n] - 1 + loc[1]),
                           origin,
                           length_tot + np.sqrt((neighbors[1] - loc[1]) ** 2 + (neighbors[0] - loc[0]) ** 2),
                           width_vals,
                           direction_masks)

        if N == 0:
            for dim0 in range(-1, 2):
                for dim1 in range(-1, 2):
                    if ((loc[1] + dim1, loc[0] + dim0) in self.ends) or (
                            (loc[1] + dim1, loc[0] + dim0) in self.branches):
                        width_vals.append(self.img_dist[loc[0] + dim0, loc[1] + dim1])
                        self.G.add_edge(origin,
                                        (loc[1] + dim1, loc[0] + dim0),
                                        length=length_tot + np.sqrt(dim1 ** 2 + dim0 ** 2),
                                        widths=width_vals)

    def combine_near_nodes_eculid(self):
        print('\t - Combining nodes which are within %d microns of each other (spacial)' % self.near_node_tol)
        for n1 in self.G.nodes():
            for n2 in self.G.nodes():
                if n1 != n2 and euclid_dist_between_nodes(n1, n2) < self.near_node_tol:
                    if n1 in self.G and n2 in self.G:
                        self.G = nx.contracted_nodes(self.G, n1, n2)

    def average_dupedge_lengths(self):
        print('\t - Cleaning edge lengths')
        for e in self.G.edges.data():
            e[2]['length'] = np.mean(e[2]['length'])

    def combine_near_nodes_length(self):
        print('\t - Combining nodes which are within %d microns of each other (path length)' % self.length_tol)
        short_list = []
        for e in self.G.edges.data():
            if e[2]['length'] <= self.length_tol:
                short_list.append((e[0], e[1]))
        for i in range(0, len(short_list)):
            if short_list[i][0] in self.G and short_list[i][1] in self.G:
                self.G = nx.contracted_nodes(self.G, short_list[i][0], short_list[i][1])

    def remove_zero_length_edges(self):
        print('\t - Removing edges of length zero')
        rem_list = []
        for e in self.G.edges.data():
            if e[2]['length'] == 0:
                rem_list.append((e[0], e[1]))

        for i in range(0, len(rem_list)):
            self.G.remove_edge(rem_list[i][0], rem_list[i][1])

    def remove_small_subgraphs(self):
        graphs = list(nx.connected_components(self.G))
        for g in graphs:
            if len(g) < self.min_nodes:
                for node in g:
                    self.G.remove_node(node)

    def get_pos_dict(self):
        position_dic = {}
        for n in self.G.nodes():
            position_dic[n] = n

        self.pos = position_dic

    def show_graph(self, with_pos=None, with_background=np.zeros(0), img_dim=None, with_skel=np.zeros(0)):
        plt.figure(figsize=(20, 20))
        edge_color = 'k'
        font_color = 'k'
        if len(with_background) > 0:
            im = with_background
            img_new_dim = im.shape
            if img_dim is not None:
                img_new_dim = img_dim
            if len(with_skel) > 0:
                img_new_dim = with_skel.shape
            im = transform.resize(im, img_new_dim)
            plt.imshow(im, cmap='gray')
            edge_color = 'w'
            font_color = 'w'
        if len(with_skel) > 0:
            plt.imshow(with_skel, cmap='Blues', alpha=0.5)

        nx.draw(self.G, pos=with_pos, with_labels=True, edge_color=edge_color, font_color=font_color,
                font_weight='bold')
        plt.show()
