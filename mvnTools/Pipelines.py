import os
import pickle

import numpy as np
# import pymesh
from matplotlib import pyplot as plt
from skimage import transform, filters
from skimage.morphology import dilation

# from mvnTools import BinaryImageTools as bit
from mvnTools import Display as d
# from mvnTools import Mesh as m
# from mvnTools import MeshTools2d as mt2
from mvnTools import Network2dTools as nw2
from mvnTools import SegmentTools2d as st2
from mvnTools.TifStack import TifStack as ts

contrasts = {'original': 0, 'rescale': 1, 'equalize': 2, 'adaptive': 3}


class Pipelines:

    def __init__(self):
        pass


def std_2d_segment(tif_file, scale_xy, scale_z=1, h_pct=1,
                   max_dim=512, max_micron=750,
                   contrast_method='rescale', thresh_method='entropy', rwalk_thresh=(0, 0),
                   bth_k=3, wth_k=3, dila_gauss=0.333,
                   open_first=False, open_k=0, close_k=5,
                   connected2D=False,
                   smooth=1, all_plots=True, review_plot=True,
                   output_dir='', name='',
                   save_mask=False, save_skel=False, save_dist=False, save_disp=False, save_review=False,
                   generate_mesh_25=True,
                   connected_mesh=True, connected_vol=False,
                   plot_25d=True, save_volume_mask=True, save_surface_meshes=True, generate_volume_meshes=False):
    img_original = ts(tif_file, max_dim=max_dim, page_list=False, flat=True)
    img_original_flat = img_original.get_flat()

    img_dim = img_original_flat.shape
    x_micron = img_original.downsample_factor * scale_xy[0] * img_dim[0]
    y_micron = img_original.downsample_factor * scale_xy[1] * img_dim[1]
    units = np.ceil(max(x_micron, y_micron) / max_micron)
    img_original.set_units(units)
    scale_xy = (
        img_original.downsample_factor * scale_xy[0] / units, img_original.downsample_factor * scale_xy[1] / units)

    expt_z = len(img_original.get_pages()) * scale_z / units
    print('NumPages: %d' % len(img_original.get_pages()))

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

    if connected2D:
        img_mask = st2.get_largest_connected_region(img_mask, plot=all_plots)

    if scale_xy[0] != 1 or scale_xy[1] != 1:
        print(' - Scaling to %d um / pixel' % units)
        current_dim = img_mask.shape
        new_dim = (np.round(current_dim[0] * scale_xy[0]), np.round(current_dim[1] * scale_xy[1]))
        img_mask = transform.resize(img_mask, new_dim)
        img_enhanced = transform.resize(img_enhanced, new_dim)

    print('\t - Smoothing mask to reduce skeleton error')
    img_mask = filters.gaussian(img_mask, sigma=(smooth * scale_xy[0] / 3.0, smooth * scale_xy[1] / 3.0),
                                preserve_range=True)

    img_mask = img_mask > np.mean(img_mask)

    print('\t\t - Using smooth=' + str(smooth) +
          ' coresponding to a gaussian filter of sigma=' +
          str('(%0.4f, %0.4f)' % (smooth * scale_xy[0] / 3.0, smooth * scale_xy[0] / 3.0)))

    print(' - Producing computationally useful transforms')
    img_skel = st2.skeleton(img_mask, plot=all_plots)
    img_dist = st2.distance_transform(img_mask, plot=all_plots)

    if all_plots or review_plot:
        d.review_2d_results(img_enhanced, img_mask, dilation(img_skel), img_dist, units=units)

    if save_mask:
        if not os.path.isdir(output_dir + 'masks2D/'):
            os.mkdir(output_dir + 'masks2D/')
        plt.imsave(output_dir + 'masks2D/' + name + ('-%dum-pix' % units) + '.png', img_mask, cmap='gray')

    if save_skel:
        if not os.path.isdir(output_dir + 'skels2D/'):
            os.mkdir(output_dir + 'skels2D/')
        plt.imsave(output_dir + 'skels2D/' + name + ('-%dum-pix' % units) + '.png', img_skel, cmap='gray')

    if save_dist:
        if not os.path.isdir(output_dir + 'distance2D/'):
            os.mkdir(output_dir + 'distance2D/')
        plt.imsave(output_dir + 'distance2D/' + name + ('-%dum-pix' % units) + '.png', img_dist, cmap='gray')

    if save_disp:
        if not os.path.isdir(output_dir + 'display/'):
            os.mkdir(output_dir + 'display/')
        plt.imsave(output_dir + 'display/' + name + ('-%dum-pix' % units) + '.png', img_enhanced, cmap='gray')

    if save_review:
        if not os.path.isdir(output_dir + 'review-segment2D/'):
            os.mkdir(output_dir + 'review-segment2D/')
        d.review_2d_results(img_enhanced, img_mask, dilation(img_skel), img_dist,
                            saving=True, units=units).savefig(
            output_dir + 'review-segment2D/' + name + ('-%dum-pix' % units) + '.png')
        plt.show(block=False)
        plt.close()

    # if generate_mesh_25:
    #     segment_2d_to_meshes(img_dist, img_skel, units=units, h_pct=h_pct, expt_z=expt_z, connected_mesh=connected_mesh,
    #                          connected_vol=connected_vol,
    #                          plot_25d=plot_25d,
    #                          save_volume_mask=save_volume_mask,
    #                          save_surface_meshes=save_surface_meshes, generate_volume_meshes=generate_volume_meshes,
    #                          output_dir=output_dir, name=name)

    return img_enhanced, img_mask, img_skel, img_dist, img_original

# def segment_2d_to_meshes(img_dist, img_skel, units=1, h_pct=1, expt_z=0, connected_mesh=True, connected_vol=False,
#                          plot_25d=True,
#                          save_volume_mask=True,
#                          save_surface_meshes=True, generate_volume_meshes=False,
#                          output_dir='', name=''):
#     img_dist = mt2.smooth_dtransform_auto(img_dist, img_skel)
#     img_dist = np.pad(img_dist, 1, 'constant', constant_values=0)
#     if h_pct < 0:
#         h_pct = (expt_z / 2) / np.amax(img_dist)
#         print('\t\t - Calculated h_pct %.4f' % h_pct)
#     img_dist = img_dist * h_pct
#     img_3d = mt2.img_dist_to_img_volume(img_dist)
#     img_3d = np.append(img_3d, np.flip(img_3d, axis=0), axis=0)
#
#     if save_volume_mask:
#         img_3d_save = img_3d.copy()
#         if connected_vol:
#             img_3d_save = st2.get_largest_connected_region(img_3d_save, plot=False, verbose=False)
#         bit.volume_density_data(img_3d_save, units=units, output_dir=output_dir, name=name,
#                                 connected=connected_vol, source='25d', save_ims=True)
#     if connected_mesh:
#         img_3d = st2.get_largest_connected_region(img_3d, plot=False, verbose=False)
#
#     mesh, verts = m.generate_surface(img_3d, iso=0, grad='ascent',
#                                      plot=plot_25d, connected=connected_mesh, clean=True, offscreen=False)
#
#     if save_surface_meshes:
#         if not os.path.isdir(output_dir + 'surface-meshes/'):
#             os.mkdir(output_dir + 'surface-meshes/')
#
#         filepath = output_dir + 'surface-meshes/' + name + '-25d' + ('-%dum-pix' % units)
#         pymesh.save_mesh(filepath + '.obj', mesh)
#
#     if generate_volume_meshes:
#         if not os.path.isdir(output_dir + 'volume-meshes/'):
#             os.mkdir(output_dir + 'volume-meshes/')
#
#         p1 = output_dir + 'surface-meshes/' + name + '-25d' + ('-%dum-pix' % units) + '.obj'
#         if os.path.isfile(p1):
#             m.generate_lumen_tetmsh(p1, path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-25d' + (
#                     '-%dum-pix' % units) + '.msh',
#                                     removeOBJ=False)
#         else:
#             pymesh.save_mesh(p1, mesh)
#             m.generate_lumen_tetmsh(p1, path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-25d' + (
#                     '-%dum-pix' % units) + '.msh',
#                                     removeOBJ=True)
#         m.create_ExodusII_file(output_dir + 'volume-meshes/' + name + '-25d' + ('-%dum-pix' % units) + '.msh',
#                                path_to_e='', removeMSH=False)


def network_2d_analysis(G=None, path_to_G='',
                        output_dir='', name='', save_outs=True, plot_outs=False):
    if G is None and len(path_to_G) > 0:
        with open(path_to_G, 'rb') as network_object:
            G = pickle.load(network_object)

    nw2.get_directionality_pct_pix(G.img_dir, img_dist=G.img_dist, weighted=False,
                                   outputdir=output_dir, save_vals=save_outs)

    nw2.plot_directionality(G.img_dir, G.img_enhanced, num_subsections=10,
                            img_dist=G.img_dist, weighted=False, output_dir=output_dir, show=plot_outs,
                            save_plot=save_outs)

    nw2.get_directionality_pct_pix(G.img_dir, img_dist=G.img_dist, weighted=True,
                                   outputdir=output_dir, save_vals=save_outs)

    nw2.plot_directionality(G.img_dir, G.img_enhanced, num_subsections=10,
                            img_dist=G.img_dist, weighted=True, output_dir=output_dir, show=plot_outs,
                            save_plot=save_outs)

    if save_outs:
        nw2.network_summary(G, output_dir)
        # nw2.network_summary_csv(G, output_dir)
    nw2.network_histograms([G.lengths], 'Segment length $(\mu m)$', 'Frequency', 'Branch Length Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.surfaces], 'Segment surface area $(\mu m)^2$', 'Frequency', 'Surface Area Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.volumes], 'Segment volume $(\mu m)^3$', 'Frequency', 'Branch Volume Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.radii], 'Segment radius $(um)$', 'Frequency', 'Branch radii Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.contractions], 'Segment contraction factor $\left(\\frac{displacement}{length}\\right)$',
                           'Frequency',
                           'Branch Contraction Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.fractal_scores],
                           'Segment fractal dimension $\left(\\frac{ln(length)}{ln(displacement)}\\right)$',
                           'Frequency',
                           'Branch Fractal Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.connectivity],
                           'Connectivity',
                           'Frequency',
                           'Node Connectivity Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
