import os
import pickle

import numpy as np
import pymesh
from matplotlib import pyplot as plt
from skimage import transform, filters
from skimage.external import tifffile as tif
from skimage.morphology import dilation, closing

from mvnTools import BinaryImageTools as bit
from mvnTools import Display as d
from mvnTools import Mesh as m
from mvnTools import MeshTools2d as mt2
from mvnTools import Network2dTools as nw2
from mvnTools import SegmentTools2d as st2
from mvnTools import SegmentTools3d as st3
from mvnTools.TifStack import TifStack as ts

contrasts = {'original': 0, 'rescale': 1, 'equalize': 2, 'adaptive': 3}


class Pipelines:
    def __init__(self):
        pass


def std_2d_segment(tif_file, scale_xy,
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
    img_original = ts(tif_file, page_list=False, flat=True)
    img_original_flat = img_original.get_flat()

    img_dim = img_original_flat.shape
    x_micron = img_original.downsample_factor * scale_xy[0] * img_dim[0]
    y_micron = img_original.downsample_factor * scale_xy[1] * img_dim[1]
    units = np.ceil(max(x_micron, y_micron) / 750)
    img_original.set_units(units)
    scale_xy = (
        img_original.downsample_factor * scale_xy[0] / units, img_original.downsample_factor * scale_xy[1] / units)

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
          str((smooth * scale_xy[0] / 3.0, smooth * scale_xy[0] / 3.0)))

    print(' - Producing computationally useful transforms')
    img_skel = st2.skeleton(img_mask, plot=all_plots)
    img_dist = st2.distance_transform(img_mask, plot=all_plots)

    if all_plots or review_plot:
        d.review_2d_results(img_enhanced, img_mask, dilation(img_skel), img_dist)

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
                            saving=True).savefig(
            output_dir + 'review-segment2D/' + name + ('-%dum-pix' % units) + '.png')
        plt.show(block=False)
        plt.close()

    if generate_mesh_25:
        segment_2d_to_meshes(img_dist, img_skel, units=units, connected_mesh=connected_mesh,
                             connected_vol=connected_vol,
                             plot_25d=plot_25d,
                             save_volume_mask=save_volume_mask,
                             save_surface_meshes=save_surface_meshes, generate_volume_meshes=generate_volume_meshes,
                             output_dir=output_dir, name=name)

    return img_enhanced, img_mask, img_skel, img_dist, img_original


def segment_2d_to_meshes(img_dist, img_skel, units=1, connected_mesh=True, connected_vol=False,
                         plot_25d=True,
                         save_volume_mask=True,
                         save_surface_meshes=True, generate_volume_meshes=False,
                         output_dir='', name=''):
    img_dist = mt2.smooth_dtransform_auto(img_dist, img_skel)
    img_dist = np.pad(img_dist, 1, 'constant', constant_values=0)
    img_3d = mt2.img_dist_to_img_volume(img_dist)
    img_3d = np.append(img_3d, np.flip(img_3d, axis=0), axis=0)

    if save_volume_mask:
        img_3d_save = img_3d.copy()
        if connected_vol:
            img_3d_save = st2.get_largest_connected_region(img_3d_save, plot=False, verbose=False)
        bit.volume_density_data(img_3d_save, units=units, output_dir=output_dir, name=name,
                                connected=True, source='25d', save_ims=True)

    if connected_mesh:
        img_3d = st2.get_largest_connected_region(img_3d, plot=False, verbose=False)

    mesh, verts = m.generate_surface(img_3d, iso=0, grad='ascent',
                                     plot=plot_25d, connected=connected_mesh, clean=True, offscreen=False)

    if save_surface_meshes:
        if not os.path.isdir(output_dir + 'surface-meshes/'):
            os.mkdir(output_dir + 'surface-meshes/')

        filepath = output_dir + 'surface-meshes/' + name + '-25d' + ('-%dum-pix' % units)
        pymesh.save_mesh(filepath + '.obj', mesh)

    if generate_volume_meshes:
        if not os.path.isdir(output_dir + 'volume-meshes/'):
            os.mkdir(output_dir + 'volume-meshes/')

        p1 = output_dir + 'surface-meshes/' + name + '-25d' + ('-%dum-pix' % units) + '.obj'
        if os.path.isfile(p1):
            m.generate_lumen_tetmsh(p1, path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-25d' + (
                    '-%dum-pix' % units) + '.msh',
                                    removeOBJ=False)
        else:
            pymesh.save_mesh(p1, mesh)
            m.generate_lumen_tetmsh(p1, path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-25d' + (
                    '-%dum-pix' % units) + '.msh',
                                    removeOBJ=True)
        m.create_ExodusII_file(output_dir + 'volume-meshes/' + name + '-25d' + ('-%dum-pix' % units) + '.msh',
                               path_to_e='', removeMSH=False)


def network_2d_analysis(G=None, path_to_G='', mask_maybe_connected=None, mask_connected=None,
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

    nw2.network_summary(G, output_dir)
    nw2.network_histograms([G.lengths], 'Segment length (um)', 'Frequency', 'Branch Length Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.surfaces], 'Segment surface area (um^2)', 'Frequency', 'Surface Area Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.volumes], 'Segment volume (um^3)', 'Frequency', 'Branch Volume Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.radii], 'Segment radius (um)', 'Frequency', 'Branch radii Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.contractions], 'Segment contraction factor', 'Frequency',
                           'Branch Contraction Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)
    nw2.network_histograms([G.fractal_scores], 'Segment fractal dimension', 'Frequency', 'Branch Fractal Distribution',
                           [name], save=save_outs, ouput_dir=output_dir, show=plot_outs)


def std_3d_segment(img_2d_stack, img_mask, scale, units=1, slice_contrast='original', pre_thresh='sauvola',
                   maxr=15, minr=1, h_pct_r=0.75,
                   thresh_pct=0.15, ellipsoid_method='octants', thresh_num_oct=5, thresh_num_oct_op=3, max_iters=1,
                   n_theta=4, n_phi=4, ray_trace_mode='uniform', theta='xy', max_escape=1, path_l=1,
                   bth_k=3, wth_k=3, window_size=(7, 15, 15),
                   enforce_circular=True, h_pct_ellip=1,
                   fill_lumen_meshing=True, max_meshing_iters=3,
                   squeeze_skel_blobs=True, remove_skel_surf=True, surf_tol=5, close_skels=False, connected_3D=True,
                   plot_slices=False, plot_lumen_fill=True, plot_3d=True,
                   output_dir='', name='',
                   save_3d_masks=True,
                   generate_meshes=True, generate_skels=True,
                   plot_skels=True, save_skel=True,
                   save_surface_3D=True, generate_volume_3D=False,
                   save_surface_round=True, generate_volume_round=False):
    print('\n============')
    print('3d analysis:')
    print('============')


    expected_z_depth = len(img_2d_stack) * scale[2] / units
    scale = (scale[0] / units, scale[1] / units, scale[2] / units)

    slice_contrast = contrasts[slice_contrast]
    img_binary_array = st3.img_2d_stack_to_binary_array(img_2d_stack, bth_k=bth_k, wth_k=wth_k, plot_all=plot_slices,
                                                        window_size=window_size, slice_contrast=slice_contrast,
                                                        pre_thresh=pre_thresh)

    img_3d = img_binary_array.astype('int')
    img_3d, img_mask = st3.pre_process_fill_lumen(img_3d, img_mask)
    lumen_mask_e = st3.fill_lumen_ellipsoid(img_3d, img_mask,
                                            maxr=maxr, minr=minr, h_pct_r=h_pct_r,
                                            thresh_pct=thresh_pct, method=ellipsoid_method,
                                            thresh_num_oct=thresh_num_oct,
                                            thresh_num_oct_op=thresh_num_oct_op,
                                            max_iters=max_iters)
    img_3d_e = img_3d + lumen_mask_e
    img_3d_e, img_mask = st3.pre_process_fill_lumen(img_3d_e, img_mask)
    lumen_mask_r = st3.fill_lumen_ray_tracing(img_3d_e, img_mask,
                                              n_theta=n_theta, n_phi=n_phi, mode=ray_trace_mode, theta=theta,
                                              max_escape=max_escape, path_l=path_l)
    img_3d_r = img_3d_e + lumen_mask_r
    if connected_3D:
        img_3d_r = st2.get_largest_connected_region(img_3d_r, plot=False, verbose=False)

    if plot_lumen_fill:
        st3.show_lumen_fill(img_3d, lumen_mask_e, l2_fill=lumen_mask_r)

    img_3d_r = img_3d_r > 0
    img_3d_r = st3.scale_and_fill_z(img_3d_r, scale[2], units=units)

    if scale[0] != 1 or scale[1] != 1:
        print(' - Scaling to %d um / pixel in xy' % units)
        new_img = []
        current_dim = img_3d_r[0].shape
        new_dim = (np.round(current_dim[0] * scale[0]), np.round(current_dim[1] * scale[1]))
        for im_plane in img_3d_r:
            new_plane = transform.resize(im_plane, new_dim, preserve_range=True)
            new_plane = filters.gaussian(new_plane, sigma=(scale[0] / 3.0, scale[1] / 3.0), preserve_range=True)
            mean = np.mean(new_plane)
            new_img.append(new_plane > np.mean(mean))

        img_3d_r = np.asarray(new_img)

    img_3D = np.pad(img_3d_r, 1)
    img_round = None

    if fill_lumen_meshing:
        mesh, vert_list = m.generate_surface(img_3D, iso=0, grad='ascent', plot=False, offscreen=True,
                                             fill_internals=True)
        img_3D = st3.iterative_lumen_mesh_filling(img_3D, vert_list, max_meshing_iters)

    if enforce_circular or save_surface_round:
        if h_pct_ellip > 0:
            img_round = st3.enforce_circular(img_3D, h_pct=h_pct_ellip)
        else:
            img_probe = st3.enforce_circular(img_3D, h_pct=1)
            h_pct_ellip = expected_z_depth / img_probe.shape[0]
            img_round = st3.enforce_circular(img_3D, h_pct_ellip)
        img_round = np.pad(img_round, 1)

        if fill_lumen_meshing:
            mesh, vert_list = m.generate_surface(img_round, iso=0, grad='ascent', plot=False, offscreen=True,
                                                 fill_internals=True)
            img_round = st3.iterative_lumen_mesh_filling(img_round, vert_list, max_meshing_iters)

    if generate_meshes:
        std_3d_mesh(img_3D=img_3D, img_round=img_round, units=units, h_pct_ellip=h_pct_ellip,
                    save_surface_3D=save_surface_3D, save_surface_round=save_surface_round,
                    generate_volume_3D=generate_volume_3D, generate_volume_round=generate_volume_round,
                    output_dir=output_dir, name=name, plot_3d=plot_3d)

    skel_3D, skel_round = None, None
    if generate_skels:
        skel_3D, skel_round = std_3d_skel(img_3D=img_3D, img_round=img_round, units=1, h_pct_ellip=h_pct_ellip,
                                          squeeze_blobs=squeeze_skel_blobs, remove_surfaces=remove_skel_surf,
                                          skel_close=close_skels,
                                          output_dir=output_dir, name=name, plot_3d=plot_skels, surf_tol=surf_tol,
                                          save_skel=save_skel)

    if save_3d_masks:
        bit.volume_density_data(img_3D, units=units, output_dir=output_dir, name=name, connected=connected_3D,
                                source='3D', save_ims=True)
        if img_round is not None:
            bit.volume_density_data(img_3D, units=units, output_dir=output_dir, name=name, connected=connected_3D,
                                    source='enforce-ellip-' + str('%0.4f' % h_pct_ellip),
                                    save_ims=True)

    return img_3D, img_round, skel_3D, skel_round


def std_3d_mesh(img_3D=None, img_round=None, units=1, connected=True, h_pct_ellip=0,
                save_surface_3D=True, save_surface_round=True,
                generate_volume_3D=False, generate_volume_round=False,
                output_dir='', name='', plot_3d=True):
    if img_3D is not None:
        mesh_3D, vert_list = m.generate_surface(img_3D, iso=0, grad='ascent', plot=plot_3d, offscreen=False,
                                                connected=connected,
                                                fill_internals=False, clean=True,
                                                title="'Honest' scaling (no extrapolation)")

        if save_surface_3D or generate_volume_3D:
            if not os.path.isdir(output_dir + 'surface-meshes/'):
                os.mkdir(output_dir + 'surface-meshes/')
            path = output_dir + 'surface-meshes/' + name + '-3d' + ('-%dum-pix' % units)
            pymesh.save_mesh(path + '.obj', mesh_3D)

        if generate_volume_3D:
            if not os.path.isdir(output_dir + 'volume-meshes/'):
                os.mkdir(output_dir + 'volume-meshes/')
            if save_surface_3D:
                m.generate_lumen_tetmsh(output_dir + 'surface-meshes/' + name + '-3d' + ('-%dum-pix' % units) + '.obj',
                                        path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-3d' + (
                                                    '-%dum-pix' % units) + '.msh',
                                        removeOBJ=False)
            else:
                m.generate_lumen_tetmsh(output_dir + 'surface-meshes/' + name + '-3d' + ('-%dum-pix' % units) + '.obj',
                                        path_to_volume_msh=output_dir + 'volume-meshes/' + name + '-3d' + (
                                                    '-%dum-pix' % units) + '.msh',
                                        removeOBJ=True)
            m.create_ExodusII_file(output_dir + 'volume-meshes/' + name + '-3d' + ('-%dum-pix' % units) + '.msh')

    if img_round is not None:
        mesh_round, vert_list = m.generate_surface(img_round, iso=0, grad='ascent', plot=plot_3d, offscreen=False,
                                                   connected=connected,
                                                   fill_internals=False, clean=True, title="Ellipsoid enforced scaling")

        if save_surface_round or generate_volume_round:
            if not os.path.isdir(output_dir + 'surface-meshes/'):
                os.mkdir(output_dir + 'surface-meshes/')
            path = output_dir + 'surface-meshes/' + name + '-enforce-ellip-' + str('%0.4f' % h_pct_ellip) + (
                        '-%dum-pix' % units)
            pymesh.save_mesh(path + '.obj', mesh_round)

        if generate_volume_round:
            if not os.path.isdir(output_dir + 'volume-meshes/'):
                os.mkdir(output_dir + 'volume-meshes/')
            if save_surface_round:
                m.generate_lumen_tetmsh(
                    output_dir + 'surface-meshes/' + name + '-enforce-ellip-' + str('%0.4f' % h_pct_ellip) + (
                                '-%dum-pix' % units) + '.obj',
                    path_to_volume_msh=output_dir + 'volume-meshes/' + '-enforce-ellip-' + str(
                        '%0.4f' % h_pct_ellip) + ('-%dum-pix' % units) + '.msh',
                    removeOBJ=False)
            else:
                m.generate_lumen_tetmsh(
                    output_dir + 'surface-meshes/' + name + '-enforce-ellip-' + str('%0.4f' % h_pct_ellip) + (
                                '-%dum-pix' % units) + '.obj',
                    path_to_volume_msh=output_dir + 'volume-meshes/' + '-enforce-ellip-' + str(
                        '%0.4f' % h_pct_ellip) + ('-%dum-pix' % units) + '.msh',
                    removeOBJ=True)
            m.create_ExodusII_file(
                output_dir + 'volume-meshes/' + '-enforce-ellip-' + str('%0.4f' % h_pct_ellip) + (
                            '-%dum-pix' % units) + '.msh')


def std_3d_skel(img_3D=None, img_round=None, units=1, h_pct_ellip=0,
                squeeze_blobs=False, remove_surfaces=False, surf_tol=5, skel_close=False,
                output_dir='', name='', plot_3d=True, save_skel=False):
    skel_3D, skel_round = None, None
    if img_3D is not None:
        skel_3D = st3.skeleton_3d(img_3D, squeeze_blobs=squeeze_blobs, remove_surfaces=remove_surfaces,
                                  surface_tol=surf_tol)
        if skel_close:
            skel_3D = closing(skel_3D)
        if plot_3d:
            m.generate_surface(skel_3D, connected=False, clean=True, title="'Honest' scaling (no extrapolation)")

        if save_skel:
            if not os.path.isdir(output_dir + 'skels3d/'):
                os.mkdir(output_dir + 'skels3d/')
            tif.imsave(output_dir + 'skels3d/' + name + '-3D' + ('-%dum-pix' % units) + '.tif',
                       np.asarray(skel_3D, 'uint8'), bigtiff=True)

    if img_round is not None:
        skel_round = st3.skeleton_3d(img_round, squeeze_blobs=squeeze_blobs, remove_surfaces=remove_surfaces,
                                     surface_tol=surf_tol)
        if skel_close:
            skel_round = closing(skel_round)
        if plot_3d:
            m.generate_surface(skel_round, connected=False, clean=True, title="Ellipsoid enforced scaling")

        if save_skel:
            if not os.path.isdir(output_dir + 'skels3d/'):
                os.mkdir(output_dir + 'skels3d/')
            tif.imsave(output_dir + 'skels3d/' + name + '-enforce-ellip-' + str('%0.4f' % h_pct_ellip) + (
                        '-%dum-pix' % units) + '.tif',
                       np.asarray(skel_round, 'uint8'), bigtiff=True)

    return skel_3D, skel_round
