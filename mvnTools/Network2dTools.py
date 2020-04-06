import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def get_scaled_masks(img_dir, img_dist=None, weighted=False):
    u_masks = []
    for i in range(1, 5):
        mult_factor = 1
        if i == 1 or i == 3:
            mult_factor = np.sqrt(2)
        d_mask = np.asarray(img_dir == i, 'float') * mult_factor
        if weighted:
            d_mask = d_mask * img_dist

        u_masks.append(d_mask)

    return u_masks


def get_directionality_pct_mask(img_mask):
    vert = np.sum(img_mask, axis=0)
    horz = np.sum(img_mask, axis=1)

    vert_md = np.median(vert)
    horz_md = np.median(horz)
    tot_md = vert_md + horz_md

    print('Median Masking')
    print('  Percent Vertical:\t\t %.4f\n  Percent Horizontal:\t %.4f' % (vert_md / tot_md, horz_md / tot_md))


def get_directionality_pct_pix(img_dir, img_dist=None, weighted=False, outputdir='', save_vals=False):
    u_masks = get_scaled_masks(img_dir, img_dist=img_dist, weighted=weighted)
    tot = np.sum(u_masks)
    vert = np.sum(u_masks[1]) / tot
    horz = np.sum(u_masks[3]) / tot
    left = np.sum(u_masks[0]) / tot
    right = np.sum(u_masks[2]) / tot

    data = np.asarray([vert, horz, left, right])
    zscore = (np.max(data) - np.mean(data)) / np.std(data)

    print('\n2D Directionality Summary. Weighted is %r.' % weighted)
    print('  Percent Vertical:\t\t %.4f\n  Percent Horizontal:\t %.4f\n'
          '  Percent L. Diagonal:\t %.4f\n  Percent R. Diagonal:\t %.4f\n' % (vert, horz, left, right))
    print('  Z-Score:\t %.4f\n' % zscore)


def plot_directionality(img_dir, img_enhanced, num_subsections=10, img_dist=None, weighted=False, outputdir='',
                        save_plot=False):
    dim = img_enhanced.shape

    xs = int(dim[0] / num_subsections)
    ys = int(dim[1] / num_subsections)

    u_masks = get_scaled_masks(img_dir, img_dist=img_dist, weighted=weighted)

    u_fig, u_ax = plt.subplots()
    plt.imshow(img_enhanced, alpha=0.5, cmap='gray')

    for x in range(xs, img_dir.shape[0] - xs, xs):
        for y in range(ys, img_dir.shape[1] - ys, ys):
            vert_u = np.sum(u_masks[3][x - xs:x + xs + 1, y - ys:y + ys + 1])
            horz_u = np.sum(u_masks[1][x - xs:x + xs + 1, y - ys:y + ys + 1])
            left_u = np.sum(u_masks[2][x - xs:x + xs + 1, y - ys:y + ys + 1])
            right_u = np.sum(u_masks[0][x - xs:x + xs + 1, y - ys:y + ys + 1])

            tot_u = vert_u + horz_u + left_u + right_u
            vert_u, horz_u, left_u, right_u = vert_u / tot_u, horz_u / tot_u, left_u / tot_u, right_u / tot_u

            draw_size = 2 * max(xs, ys) + 1
            ellip_vert = Ellipse((y, x), draw_size * vert_u, draw_size * horz_u,
                                 alpha=0.5, linewidth=2, edgecolor='blue')
            ellip_tilt = Ellipse((y, x), draw_size * right_u, draw_size * left_u, 45,
                                 alpha=0.5, linewidth=2, edgecolor='red')

            u_ax.add_artist(ellip_vert)
            u_ax.add_artist(ellip_tilt)

    plt.show()

def closeness_vitatlity(G, img_enhanced):
    print(nx.get_edge_attributes(G, 'length'))
    print(nx.get_edge_attributes(G, 'volume'))
    print(nx.get_edge_attributes(G, 'surface'))
    cv_dic = nx.closeness_vitality(G)
    print(cv_dic)
    #cv_dic = nx.dispersion(G)
    #print(cv_dic)
    print(nx.average_shortest_path_length(G, 'length'))
    print(nx.average_shortest_path_length(G, 'volume'))
    lc_dic = nx.load_centrality(G)

    xs = []
    ys = []
    vals = []
    value_mask = np.zeros(img_enhanced.shape)
    for node in lc_dic:
        x = int(node[0])
        y = int(node[1])
        val = lc_dic[node]
        value_mask[y-10: y+11,x-10:x+11] = val
        xs.append(xs)
        ys.append(ys)
        vals.append(val)

    minval = min(vals)
    maxval = max(vals)

    plt.imshow(img_enhanced, alpha=0.5, cmap='gray')
    plt.imshow(value_mask, alpha=0.8, cmap='afmhot', vmin=minval, vmax=maxval)
    plt.show()

    xs = []
    ys = []
    vals = []
    value_mask = np.zeros(img_enhanced.shape)
    for node in cv_dic:
        x = int(node[0])
        y = int(node[1])
        val = cv_dic[node]
        if val > 0:
            value_mask[y - 10: y + 11, x - 10:x + 11] = val
            xs.append(xs)
            ys.append(ys)
            vals.append(val)

    minval = min(vals)
    maxval = max(vals)

    plt.imshow(img_enhanced, alpha=0.5, cmap='gray')
    plt.imshow(value_mask, alpha=0.8, cmap='afmhot', vmin=minval, vmax=maxval)
    plt.show()

