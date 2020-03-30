from mvnTools.TifStack import TifStack as ts
from mvnTools import SegmentTools2d as st2
from mvnTools import SegmentTools3d as st3
from mvnTools import MeshTools2d as mt2
from mvnTools import Mesh as m
from mvnTools import NetworkTools2d as nw2
from mvnTools import Display as d
import numpy as np
from skimage import transform, filters
from skimage.morphology import dilation
from timeit import default_timer as timer

contrasts = {'original': 0, 'rescale': 1, 'equalize': 2, 'adaptive': 3}


class Pipelines:
    def __init__(self, image_array):
        pass


def std_2d_segment(tif_file, scale_xy,
                   contrast_method='rescale',
                   thresh_method='entropy',
                   rwalk_thresh=(0, 0),
                   bth_k=3,
                   wth_k=3,
                   dila_gauss=1 / 3,
                   open_first=False,
                   open_k=0,
                   close_k=5,
                   connected=False,
                   smooth=1,
                   all_plots=True,
                   review_plot=True):
    img_original = ts(tif_file, page_list=False, flat=True)
    img_original_flat = img_original.get_flat()

    enhance_contrasts = st2.contrasts(img_original_flat, plot=all_plots, adpt=0.04)
    img_enhanced = enhance_contrasts[contrasts[contrast_method]]

    print('============')
    print('2d analysis:')
    print('============')

    print(' - Filtering the flattened image')
    img = st2.black_top_hat(img_enhanced, plot=all_plots, k=bth_k)
    img = st2.white_top_hat(img, plot=all_plots, k=wth_k)
    dilated = st2.background_dilation(img, gauss=dila_gauss, plot=all_plots)

    print(' - Converting to binary mask')
    if thresh_method == 'random-walk':
        low = rwalk_thresh[0]
        high = rwalk_thresh[1]
        img_mask = st2.random_walk_thresh(dilated, low, high, plot=all_plots)
    else:
        img_mask = st2.cross_entropy_thresh(dilated, plot=all_plots, verbose=False)

    print(' - Cleaning the binary mask')
    if open_first:
        img_mask = st2.open_binary(img_mask, k=open_k, plot=all_plots)
        img_mask = st2.close_binary(img_mask, k=close_k, plot=all_plots)
    else:
        img_mask = st2.close_binary(img_mask, k=close_k, plot=all_plots)
        img_mask = st2.open_binary(img_mask, k=open_k, plot=all_plots)

    if connected:
        img_mask = st2.get_largest_connected_region(img_mask, plot=all_plots)

    if scale_xy[0] != 1 or scale_xy[1] != 1:
        print(' - Scaling to 1 um / pixel')
        current_dim = img_mask.shape
        new_dim = (np.round(current_dim[0] * scale_xy[0]), np.round(current_dim[1] * scale_xy[1]))
        img_mask = transform.resize(img_mask, new_dim)
        img_enhanced = transform.resize(img_enhanced, new_dim)


    print('\t - Smoothing mask to reduce skeleton error')
    img_mask = filters.gaussian(img_mask, sigma=(smooth * scale_xy[0] / 3.0, smooth * scale_xy[0] / 3.0))
    img_mask = img_mask > 0.5
    print('\t\t - Using smooth=' + str(smooth) +
          ' coresponding to a gaussian filter of sigma=' +
          str((smooth * scale_xy[0] / 3.0, smooth * scale_xy[0] / 3.0)))

    print(' - Producing computationally useful transforms')
    img_skel = st2.skeleton(img_mask, plot=all_plots)
    img_dist = st2.distance_transform(img_mask, plot=all_plots)

    if all_plots or review_plot:
        d.review_2d_results(img_enhanced, img_mask, dilation(img_skel), img_dist)

    return img_enhanced, img_mask, img_skel, img_dist, img_original


def segment_2d_to_meshes(img_dist, img_skel, all_plots=True):

    img_dist = mt2.smooth_dtransform_auto(img_dist, img_skel)
    img_dist = np.pad(img_dist, 1, 'constant', constant_values=0)
    img_3d = mt2.img_dist_to_img_volume(img_dist)

    mesh = m.generate_surface(img_3d, iso=0, grad='ascent', plot=all_plots, offscreen=False)
    return mesh


def generate_2d_network(img_skel, img_dist,
                        near_node_tol=5, length_tol=1,
                        img_enhanced=np.zeros(0), plot=True):
    print(' \nTransforming image data to network representation (nodes and weighted edges)')

    ends, branches = nw2.get_ends_and_branches(img_skel)
    G, img_skel_erode = nw2.create_Graph_and_Nodes(ends, branches, img_skel)
    G = nw2.fill_edges(G, ends, branches, img_dist, img_skel_erode)

    G = nw2.combine_near_nodes_eculid(G, near_node_tol)
    G = nw2.average_dupedge_lengths(G)
    G = nw2.combine_near_nodes_length(G, length_tol)
    G = nw2.remove_zero_length_edges(G)

    if plot:
        pos = nw2.get_pos_dict(G)
        nw2.show_graph(G, with_pos=pos, with_background=img_enhanced, with_skel=dilation(img_skel))
        nw2.show_graph(G)

    return G


def std_3d_segment(img_2d_stack, img_mask, scale,
                   maxr=15, minr=1, h_pct_r=0.75,
                   thresh_pct=0.15, ellisoid_method='octants', thresh_num_oct=5, thresh_num_oct_op=3, max_iters=1,
                   n_theta=4, n_phi=4, ray_trace_mode='uniform', max_escape=1, path_l=1,
                   plot_slices=False, plot_lumen_fill=True, plot_3d=True):
    print('\n============')
    print('3d analysis:')
    print('============')
    img_binary_array = st3.img_2d_stack_to_binary_array(img_2d_stack, bth_k=3, wth_k=3, plot_all=plot_slices)

    img_3d = img_binary_array
    img_3d, img_mask = st3.pre_process_fill_lumen(img_3d, img_mask)
    lumen_mask_e = st3.fill_lumen_ellipsoid(img_3d, img_mask,
                                            maxr=maxr, minr=minr, h_pct_r=h_pct_r,
                                            thresh_pct=thresh_pct, method=ellisoid_method,
                                            thresh_num_oct=thresh_num_oct,
                                            thresh_num_oct_op=thresh_num_oct_op,
                                            max_iters=max_iters)
    img_3d_e = img_3d + lumen_mask_e
    img_3d_e, img_mask = st3.pre_process_fill_lumen(img_3d_e, img_mask)
    lumen_mask_r = st3.fill_lumen_ray_tracing(img_3d, img_mask,
                                              n_theta=n_theta, n_phi=n_phi, mode=ray_trace_mode, max_escape=max_escape,
                                              path_l=path_l)
    img_3d_r = img_3d_e + lumen_mask_r
    if plot_lumen_fill:
        st3.show_lumen_fill(img_3d, lumen_mask_e, l2_fill=lumen_mask_r)

    img_3d_r = img_3d_r > 0
    img_3d_r[True] = 1

    img_3d_r = st3.scale_and_fill_z(img_3d_r, scale[2])
    img_3d_r = np.pad(img_3d_r, 1)

    if scale[0] != 1 or scale[1] != 1:
        print(' - Scaling to 1 um / pixel in xy')
        new_img = []
        current_dim = img_mask.shape
        new_dim = (np.round(current_dim[0] * scale[0]), np.round(current_dim[1] * scale[1]))
        for im_plane in img_3d_r:
            new_img.append(transform.resize(im_plane, new_dim))

        img_3d_r = np.asarray(new_img)


    mesh = m.generate_surface(img_3d_r, iso=0, grad='ascent', plot=plot_3d, offscreen=False)
    skel_3d = st3.skeleton_3d(img_3d_r)
    m.generate_surface(skel_3d, iso=0, grad='ascent', plot=plot_3d, offscreen=False, connected=False)

    return img_3d_r, mesh, skel_3d
