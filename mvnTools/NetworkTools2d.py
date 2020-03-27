import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from skimage import transform, io

def euclid_dist_between_nodes(n1, n2):
    return np.sqrt((n1[0]-n2[0])**2+(n1[1]-n2[1])**2)


def get_ends_and_branches(img_skel):
    print('\t - Identifying locations of branches and ends')
    ends = []
    branches = []

    img_skel = np.asarray(img_skel, 'uint8')
    img_skel = img_skel / (np.amax(img_skel))
    dim = img_skel.shape

    for y in range(0, dim[0]):
        for x in range(0, dim[1]):

            # Get the index of skeleton point
            if img_skel[x, y] == 1:

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

    ends = set(ends)
    branches = set(branches)

    return ends, branches


def create_Graph_and_Nodes(ends, branches, img_skel):
    print('\t - Establishing graph with branch and end nodes')
    G = nx.Graph()
    img_skel_erode = img_skel.copy()

    nodeID = 0
    for b in branches:
        G.add_node(b)
        img_skel_erode[b[1], b[0]] = 0
        nodeID += 1

    for e in ends:
        G.add_node(e)
        img_skel_erode[e[1], e[0]] = 0
        nodeID += 1

    return G, img_skel_erode


def edge_walk(loc, img_skel_erode, img_dist_t, G, origin, length_tot, width_vals, end_set, branch_set):
    loc = (int(loc[0]), int(loc[1]))
    img_skel_erode[loc] = 0

    neighbors = np.where(img_skel_erode[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2])
    N = int(np.sum(img_skel_erode[loc[0] - 1:loc[0] + 2, loc[1] - 1:loc[1] + 2]))
    for n in range(0, N):
        width_vals.append(img_dist_t[loc])
        edge_walk((neighbors[0][n] - 1 + loc[0], neighbors[1][n] - 1 + loc[1]),
                  img_skel_erode,
                  img_dist_t,
                  G,
                  origin,
                  length_tot + np.sqrt((neighbors[1] - loc[1]) ** 2 + (neighbors[0] - loc[0]) ** 2),
                  width_vals,
                  end_set, branch_set)

    if N == 0:
        for dim0 in range(-1, 2):
            for dim1 in range(-1, 2):
                if ((loc[1] + dim1, loc[0] + dim0) in end_set) or ((loc[1] + dim1, loc[0] + dim0) in branch_set):
                    width_vals.append(img_dist_t[loc[0] + dim0, loc[1] + dim1])
                    G.add_edge(origin,
                               (loc[1] + dim1, loc[0] + dim0),
                               length=length_tot + np.sqrt(dim1 ** 2 + dim0 ** 2),
                               widths=width_vals)


def fill_edges(G, ends, branches, img_dist, img_skel_erode):
    print('\t - Connecting nodes with weighted edges')
    for b in branches:
        loc = (b[1], b[0])
        img_dist_t = img_dist
        origin = b
        length_tot = 0
        width_vals = []
        end_set = ends
        branch_set = branches
        edge_walk(loc, img_skel_erode, img_dist_t, G, origin, length_tot, width_vals, end_set, branch_set)

    return G


def show_graph(G, with_pos=None, with_background=np.zeros(0), img_dim=None, with_skel=np.zeros(0)):
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

    nx.draw(G, pos=with_pos, with_labels=True, edge_color=edge_color, font_color=font_color, font_weight='bold')
    plt.show()


def get_pos_dict(G):
    position_dic = {}
    for n in G.nodes():
        position_dic[n] = n

    return position_dic


def average_dupedge_lengths(G):
    print('\t - Cleaning edge lengths')
    for e in G.edges.data():
        e[2]['length'] = np.mean(e[2]['length'])

    return G

def combine_near_nodes_eculid(G, tol=5):
    print('\t - Combining nodes which are within %d microns of each other (spacial)' % tol)
    for n1 in G.nodes():
        for n2 in G.nodes():
            if n1 != n2 and euclid_dist_between_nodes(n1, n2) < tol:
                G = nx.contracted_nodes(G, n1, n2)
    return G


def remove_zero_length_edges(G):
    print('\t - Removing edges of length zero')
    rem_list = []
    for e in G.edges.data():
        if e[2]['length'] == 0:
            rem_list.append((e[0], e[1]))

    for i in range(0, len(rem_list)):
        G.remove_edge(rem_list[i][0], rem_list[i][1])

    return G


def combine_near_nodes_length(G, tol=1):
    print('\t - Combining nodes which are within %d microns of each other (path length)' % tol)
    short_list = []
    for e in G.edges.data():
        if e[2]['length'] <= tol:
            short_list.append((e[0], e[1]))
    for i in range(0, len(short_list)):
        G = nx.contracted_nodes(G, short_list[i][0], short_list[i][1])

    return G