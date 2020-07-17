import csv

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats


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
    max_score = np.max(data)

    print('\n2D Directionality Summary. Weighted is %r.' % weighted)
    print('  Percent Vertical (90 degrees):\t %.4f\n  Percent Horizontal (0 degrees):\t %.4f\n'
          '  Percent L. Diagonal (135 degrees):\t %.4f\n  Percent R. Diagonal (45 degrees):\t %.4f\n' % (
          vert, horz, left, right))
    print('  Max score:\t %.4f\n' % max_score)

    if save_vals:
        file = outputdir + 'directionality.txt'
        file = open(file, 'a')
        file.write('\n2D Directionality Summary. Weighted is %r.\n\n' % weighted)
        file.write('  Percent Vertical (90 degrees):\t %.4f\n  Percent Horizontal (0 degrees):\t %.4f\n'
                   '  Percent L. Diagonal (135 degrees):\t %.4f\n  Percent R. Diagonal (45 degrees):\t %.4f\n' % (
                   vert, horz, left, right))
        file.write('\n  Max score:\t %.4f\n' % max_score)
        file.close()


def plot_directionality(img_dir, img_enhanced, num_subsections=10, img_dist=None, weighted=False, output_dir='',
                        show=False,
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

    if save_plot:
        if weighted:
            outputfile = output_dir + 'directionality_plot-weighted.png'
        else:
            outputfile = output_dir + 'directionality_plot-unweighted.png'
        plt.savefig(outputfile)

    if show:
        plt.show()
    else:
        plt.show(block=False)
        plt.close()


def network_summary(G, output_dir):
    nwsum = open(output_dir + 'network_summary.txt', 'a')
    nwsum.write('%d\tNumber of branch points\n' % G.total_branch_points)
    nwsum.write('%d\tNumber of end points\n' % G.total_ends)
    nwsum.write('%.4f\tTotal length um\n' % G.total_length)
    nwsum.write('%.4f\tTotal surface area um^2 (assumes circular vessels)\n' % G.total_surface)
    nwsum.write('%.4f\tTotal volume um^3 (assumes circular vessels)\n' % G.total_volume)
    nwsum.write('%.4f\tAverage branch length\n' % np.mean(G.lengths))
    nwsum.write('%.4f\tAverage branch surface area\n' % np.mean(G.surfaces))
    nwsum.write('%.4f\tAverage branch volume\n' % np.mean(G.volumes))
    nwsum.write('%.4f\tAverage branch radius\n' % np.mean(G.radii))
    nwsum.write('%.4f\tAverage fractal dimension\n' % np.mean(G.fractal_scores))
    nwsum.write('%.4f\tAverage contraction factor\n\n' % np.mean(G.contractions))
    nwsum.write('%.4f\tAverage node connectivity\n\n' % np.mean(G.connectivity))
    nwsum.close()


def network_summary_csv(G, output_dir):
    with open(output_dir + 'network_summary.csv', mode='w') as net_file:
        net_writer = csv.writer(net_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        net_writer.writerow(['Number of branch points', G.total_branch_points])
        net_writer.writerow(['Number of end points', G.total_ends])
        net_writer.writerow(['Total length um', G.total_length])
        net_writer.writerow(['Total surface area um^2 (assumes circular vessels)', G.total_surface])
        net_writer.writerow(['Total volume um^3 (assumes circular vessels)', G.total_volume])
        net_writer.writerow(['Average branch length', np.mean(G.lengths)])
        net_writer.writerow(['Average branch surface area', np.mean(G.surfaces)])
        net_writer.writerow(['Average branch volume', np.mean(G.volumes)])
        net_writer.writerow(['Average branch radius', np.mean(G.radii)])
        net_writer.writerow(['Average fractal dimension', np.mean(G.fractal_scores)])
        net_writer.writerow(['Average contraction factor', np.mean(G.contractions)])
        net_writer.writerow(['Average node connectivity', np.mean(G.connectivity)])


def network_histograms(data, xlabel, ylabel, title, legend, bins=20, save=True, ouput_dir='', show=False):
    plt.rcParams.update({'font.size': 22})
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(30, 15), sharex=True,
                                   sharey=True)
    ax1.hist(data, bins=bins, density=True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title + ': Histogram')

    for d in data:
        xmin, xmax = min(d), max(d)
        lnspc = np.linspace(xmin, xmax, len(d))
        m, s = stats.norm.fit(d)
        pdf_g = stats.norm.pdf(lnspc, m, s)
        ax2.plot(lnspc, pdf_g)

    ax2.set_xlabel(xlabel)
    ax2.set_title(title + ': Normal Fit')

    plt.legend(legend)

    if save:
        plt.savefig(ouput_dir + title + '.png')

    if not show:
        plt.show(block=False)
        plt.close()
    else:
        plt.show()
