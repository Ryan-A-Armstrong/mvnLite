import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from numba import jit, njit, prange
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology, transform, filters, io
from skimage.external import tifffile as tif
from skimage.filters import threshold_li, threshold_sauvola

import mvnTools.SegmentTools2d as st2
from mvnTools.Mesh import generate_surface
from mvnTools.MeshTools2d import smooth_dtransform_auto


def interp_slices(top, bottom, percent_bottom):
    dist_top_pos = ndi.distance_transform_edt(top)
    dist_bot_pos = ndi.distance_transform_edt(bottom)

    dist_top_neg = ndi.distance_transform_edt(1 - top)
    dist_bot_neg = ndi.distance_transform_edt(1 - bottom)

    top = dist_top_pos - dist_top_neg
    bot = dist_bot_pos - dist_bot_neg

    out = ((top * (1 - percent_bottom) + bot * percent_bottom) > 0).astype('int')

    return out


def scale_and_fill_z(im_25d, z_scale):
    print(' - Interpolating and scaling z-planes to 1 um / pixel.')
    off_int = z_scale - int(z_scale)
    off_int_cum = 0

    img_3d = []
    for i in range(0, len(im_25d) - 1):
        bottom = im_25d[i + 1]
        top = im_25d[i]

        img_3d.append(top)
        num_int_img = int(z_scale - 1)
        if off_int_cum > 1:
            num_int_img += 1
            off_int_cum -= 1
        else:
            off_int_cum += off_int

        p_val = 1 / (num_int_img + 1)

        for z in range(0, num_int_img):
            img_3d.append(interp_slices(top, bottom, (z + 1) * p_val))

    img_3d = np.asarray(img_3d)
    return img_3d


def pre_process_fill_lumen(img_3d, img_mask):
    dims = img_3d.shape
    mask_dim = img_mask.shape
    if mask_dim[0] != dims[1] or mask_dim[1] != dims[2]:
        img_mask = transform.resize(img_mask, (dims[1], dims[2]))
        img_mask = filters.gaussian(img_mask, sigma=((dims[1] / mask_dim[0]) / 3.0, (dims[2] / mask_dim[1]) / 3.0))
        img_mask = img_mask > 0.5
        plt.imshow(img_mask)
        plt.show()

    return img_3d, img_mask


@njit(parallel=True)
def get_ellipsoid_filter(r, h):
    n = int(2 * r - 1)
    a, b = int(n / 2), int(n / 2)
    ball_filter_blank = np.zeros((1, n, n))
    ball_filter = ball_filter_blank.copy()

    for b_slice in range(0, h):
        mask = np.zeros((n, n))
        scaled_r = (b_slice / h) * r
        for x in prange(n):
            for y in prange(n):
                if (a - x) ** 2 + (b - y) ** 2 <= scaled_r ** 2:
                    mask[x, y] = 1

        plane = ball_filter_blank
        plane[0] = mask
        ball_filter = np.append(ball_filter, plane, axis=0)

    ball_mask = ball_filter
    for b_slice in range(1, h):
        plane = ball_filter_blank
        plane[0] = ball_filter[h - b_slice]
        ball_mask = np.append(ball_mask, plane, axis=0)

    return ball_mask[1::]



@njit(fastmath=True)
def get_new_pos(old_pos, k, path_l):
    delz = path_l * np.cos(k[0])
    delx = path_l * np.sin(k[0]) * np.cos(k[1])
    dely = path_l * np.sin(k[0]) * np.sin(k[1])

    new_pos = (old_pos[0] + delz, old_pos[1] + delx, old_pos[2] + dely)
    return new_pos


@njit(parallel=True)
def fill_lumen_ray_tracing(img_3d, img_mask,
                           n_theta=3, n_phi=3, theta='sweep', mode='uniform', max_escape=1,
                           path_l=1, skel=False):
    if skel:
        print(' - Using ray tracing to collapse skeleton blobs')
    else:
        print(' - Identifying and filling internal lumen holes via ray tracing')
    dims = img_3d.shape
    lumen_mask = np.zeros(dims, np.int8)

    dims = list(dims)
    dim_z, dim_x, dim_y = dims[0], dims[1], dims[2]

    phis = np.linspace(0, 2 * np.pi, n_phi + 1)[1::]
    if theta == 'xy':
        thetas = np.array([np.pi / 2])
    elif theta == 'exclude_z_pole':
        thetas = np.linspace(-np.pi / 2, np.pi / 2, n_theta + 1)
    else:
        thetas = np.linspace(-np.pi / 2, np.pi / 2, n_theta)

    for z in prange(0, dim_z):
        for x in prange(0, dim_x):
            for y in prange(0, dim_y):
                if img_mask[x, y] and not img_3d[z, x, y]:
                    escaped = 0
                    free_path = False

                    for i_t in prange(len(thetas)):
                        if free_path:
                            break
                        if mode == 'random':
                            t = np.random.rand() * np.pi - np.pi / 2
                        else:
                            t = thetas[i_t]

                        for i_p in prange(len(phis)):
                            if theta == 'exclude_z_pole':
                                if t == 0:
                                    break

                            if free_path:
                                break
                            if mode == 'random':
                                p = np.random.rand() * 2 * np.pi
                            else:
                                p = phis[i_p]

                            k = (t, p)
                            z_im, x_im, y_im = z, x, y
                            pos = (z_im, x_im, y_im)
                            in_img = True

                            while in_img and not img_3d[z_im, x_im, y_im]:
                                pos = get_new_pos(pos, k, path_l)
                                z_im = int(pos[0] + 0.5)
                                x_im = int(pos[1] + 0.5)
                                y_im = int(pos[2] + 0.5)

                                in_img = (0 <= z_im < dims[0]) and \
                                         (0 <= x_im < dims[1]) and \
                                         (0 <= y_im < dims[2])

                            if not in_img:  # Loop broke because ray got to image edge
                                escaped = escaped + 1
                                free_path = escaped >= max_escape

                    if not free_path:
                        lumen_mask[z, x, y] = 1

    return lumen_mask


@njit(parallel=True)
def fill_lumen_ellipsoid(img_3d, img_mask,
                         maxr=15, minr=1, h_pct_r=0.75,
                         thresh_pct=0.15, method='octants', thresh_num_oct=5, thresh_num_oct_op=3,
                         max_iters=1):
    r = maxr
    dims = img_3d.shape
    dim_z, dim_x, dim_y = dims[0], dims[1], dims[2]
    lumen_mask_tot = np.zeros(dims)

    print(' - Filling lumen using ellipsoid structuring element (this may take several minutes).')
    print('\t - Using xy radius of:')
    while r >= minr:
        print(r)

        num_changes, its = 1, 0
        h = int(r * h_pct_r)
        ball_mask = get_ellipsoid_filter(r, h)

        if method == 'octants' or method == 'pairs':
            oct_max_score = np.sum(ball_mask[0:h + 1, 0:r + 1, 0:r + 1])
            thresh = thresh_pct * oct_max_score

        elif method == 'ball':
            ball_max_score = np.sum(ball_mask)
            thresh = thresh_pct * ball_max_score

        while num_changes > 0 and its < max_iters:
            its = its + 1

            lumen_mask = np.zeros(dims, np.int8)
            for z in prange(h, dim_z - h):
                for x in prange(r, dim_x - r):
                    for y in prange(r, dim_y - r):
                        if img_mask[x, y] and not img_3d[z, x, y]:
                            cube_region = img_3d[z - h + 1:z + h, x - r + 1:x + r, y - r + 1:y + r]
                            filtered_cube = cube_region * ball_mask
                            total_hot = np.sum(filtered_cube)

                            if method == 'ball':
                                if total_hot >= thresh:
                                    lumen_mask[z, x, y] = 1

                            if method == 'octants' or method == 'pairs':
                                oct_count = 0
                                oct_op_pairs = 0

                                octants = np.zeros(8)
                                octants[0] = np.sum(filtered_cube[0:h + 1, 0:r + 1, 0:r + 1])
                                octants[1] = np.sum(filtered_cube[h:2 * h + 1, r:2 * r + 1, r:2 * r + 1])

                                octants[2] = np.sum(filtered_cube[0:h + 1, r:2 * r + 1, 0:r + 1])
                                octants[3] = np.sum(filtered_cube[h:2 * h + 1, 0:r + 1, r:2 * r + 1])

                                octants[4] = np.sum(filtered_cube[0:h + 1, 0:r + 1, r:2 * r + 1])
                                octants[5] = np.sum(filtered_cube[h:2 * h + 1, r:2 * r + 1, 0:r + 1])

                                octants[6] = np.sum(filtered_cube[h:2 * h + 1, 0:r + 1, 0:r + 1])
                                octants[7] = np.sum(filtered_cube[0:h + 1, r:2 * r + 1, r:2 * r + 1])

                                for oc in prange(8):
                                    oct_count += octants[oc] > thresh
                                    if oc % 2 == 0:
                                        oct_op_pairs += octants[oc] > thresh and octants[oc + 1] > thresh

                                if method == 'octants':
                                    if oct_count > thresh_num_oct:
                                        lumen_mask[z, x, y] = 1

                                if method == 'pairs':
                                    if oct_op_pairs > thresh_num_oct_op:
                                        lumen_mask[z, x, y] = 1

            num_changes = np.sum(lumen_mask)
            img_3d = img_3d + lumen_mask
            lumen_mask_tot = lumen_mask_tot + lumen_mask
        r = r - 1

    return lumen_mask_tot


def show_lumen_fill(img_25d, l_fill, l2_fill=None, cmap1='Blues', cmap2='Reds'):
    dims = img_25d.shape
    tiff_stack = []

    for p in range(0, dims[0]):
        fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
        ax[0].set_title('Original')
        ax[0].imshow(img_25d[p], cmap='gray')

        ax[1].set_title('Lumen Mask')
        alpha = 1
        if l2_fill is not None:
            alpha = 0.5
            ax[1].imshow(l2_fill[p], cmap=cmap2, alpha=alpha)
        ax[1].imshow(l_fill[p], cmap=cmap1, alpha=alpha)

        ax[2].set_title('Original + Lumen Mask')
        ax[2].imshow(img_25d[p], cmap='gray')
        ax[2].imshow(l_fill[p], cmap=cmap1, alpha=0.5 * alpha)
        if l2_fill is not None:
            ax[2].imshow(l2_fill[p], cmap=cmap2, alpha=0.5 * alpha)

        plt.savefig('tmppage.png')
        tiff_stack.append(io.imread('tmppage.png'))
        plt.show(block=False)
        plt.close()

    tif.imsave('tmptiff.tiff', np.asarray(tiff_stack), bigtiff=True)
    t = tif.imread('tmptiff.tiff')
    tif.imshow(t)
    plt.show()
    os.remove('tmppage.png')
    os.remove('tmptiff.tiff')
    plt.close()


@jit(parallel=True)
def fill_lumen_meshing(img_3d, vert_list):
    for v in prange(0, len(vert_list)):
        mins = np.amin(vert_list[v], axis=0)
        maxes = np.amax(vert_list[v], axis=0)

        z_min, z_max = int(mins[0]), int(maxes[0])
        x_min, x_max = int(mins[1]), int(maxes[1])
        y_min, y_max = int(mins[2]), int(maxes[2])

        img_3d[z_min:z_max + 1, x_min:x_max + 1, y_min:y_max + 1] = 1

    return img_3d


def iterative_lumen_mesh_filling(img_3d, vert_list, max_iters):
    print('Filling lumen holes by iteratively meshing')
    its = 0
    while len(vert_list) > 0 and its < max_iters:
        img_3d = fill_lumen_meshing(img_3d, vert_list)
        mesh, vert_list = generate_surface(img_3d, connected=True, clean=True, fill_internals=True, plot=False)
        its += 1

    return img_3d


def enforce_circular(img_3d_binary, h_pct=1):
    print(' - Enforcing ellipsoid lumens with height = %.3f*radius_xy' % h_pct)
    dist_stack = []
    for b_plane in img_3d_binary:
        d_plane = ndi.distance_transform_edt(b_plane)
        s_plane = morphology.skeletonize(b_plane)
        d_plane = np.asarray(d_plane) * h_pct
        s_plane = np.asarray(s_plane, 'int')
        smoothed_d_plane = smooth_dtransform_auto(d_plane, s_plane, verbose=False)
        dist_stack.append(smoothed_d_plane)

    dist_array = np.asarray(dist_stack)
    worst_case_pad = int(np.amax(dist_array) + 0.5)
    dist_array = np.pad(dist_array, ((worst_case_pad,), (0,), (0,)))

    return fill_circular(dist_array, worst_case_pad)


@njit(parallel=True)
def fill_circular(padded_dist_array, pad_r):
    dim = padded_dist_array.shape
    img_3d_circ = np.zeros(dim)
    Z = dim[0]

    for z in prange(pad_r, Z - pad_r):
        maxr = int(np.ceil(np.amax(padded_dist_array[z])))

        for z_slice in prange(0, maxr):
            filtered_plane = (maxr - np.ceil(padded_dist_array[z])) <= z_slice
            img_3d_circ[z + (maxr - z_slice)] = img_3d_circ[z + (maxr - z_slice)] + filtered_plane
            img_3d_circ[z - (maxr - z_slice)] = img_3d_circ[z - (maxr - z_slice)] + filtered_plane

    return img_3d_circ > 0

def enforce_circular2(img_3d_binary, h_pct=1):
    print(' - Enforcing ellipsoid lumens with height = %.3f*radius_xy' % h_pct)
    dist_3d = distance_transform_edt(img_3d_binary)
    skel_3d = skeleton_3d(img_3d_binary)

    dist_array = np.asarray(dist_3d)*h_pct
    worst_case_pad = int(np.amax(dist_array) + 0.5)
    dist_array = np.pad(dist_array, ((worst_case_pad,), (0,), (0,)))
    skel_3d = np.pad(skel_3d, ((worst_case_pad,), (0,), (0,)))

    return fill_circular2(dist_array, worst_case_pad, skel_3d)


@njit(parallel=False)
def fill_circular2(padded_dist_array, pad_r, skel_3d):
    dim = padded_dist_array.shape
    img_3d_circ = np.zeros(dim)

    Z = dim[0]
    X = dim[1]
    Y = dim[2]

    for z in prange(0, Z):
        for x in prange(0, X):
            for y in prange(0, Y):
                if skel_3d[z, x, y]:
                    r = int(padded_dist_array[z, x, y]+0.5)
                    filler = get_ellipsoid_filter(r, r)
                    img_3d_circ[z-r: z+r+1, x-r:x+r+1, y-r: y+r+1] += filler

    return img_3d_circ > 0


def img_2d_stack_to_binary_array(img_2d_stack, bth_k=3, wth_k=3, close_k=1, plot_all=False, window_size=(7, 15, 15),
                                 slice_contrast=0, pre_thresh='sauvola'):
    print(' - Converting each z-plane to a binary mask')
    img_2d_stack = np.asarray(img_2d_stack)
    img_stacked = []
    for im_plane in range(0, len(img_2d_stack)):
        im = img_2d_stack[im_plane]
        if slice_contrast:
            im = st2.contrasts(im, plot=False)[slice_contrast]
        im = st2.black_top_hat(im, plot=False, k=bth_k, verbose=False)
        im = st2.white_top_hat(im, plot=False, k=wth_k, verbose=False)
        img_stacked.append(im)

    img_out = []
    if pre_thresh != 'none':
        thresh_3d = threshold_sauvola(np.asarray(img_stacked), window_size=window_size, k=0.3)
    else:
        thresh_3d = np.asarray(img_stacked)

    for plane in range(0, len(thresh_3d)):
        thresh = thresh_3d[plane] > threshold_li(thresh_3d[plane])
        img_out.append(st2.close_binary(thresh, k=close_k, plot=False, verbose=False))

    img_out = np.asarray(img_out)
    img_out = img_out / np.amax(img_out)
    img_out = morphology.binary_opening(img_out)
    img_out = morphology.binary_closing(img_out)
    img_out = st2.get_largest_connected_region(img_out, plot=False, verbose=False)
    if plot_all:
        tif.imsave('tmp.tiff', np.asarray(img_out, 'uint8'), bigtiff=True)
        t = tif.imread('tmp.tiff')
        tif.imshow(t)
        plt.show()
        os.remove('tmp.tiff')

    return img_out


def skeleton_3d(img_3d, squeeze_blobs=False, remove_surfaces=False, surface_tol=5):
    print(' - Thinning to 3D skeleton.')
    warnings.filterwarnings('ignore')
    skel_3d = morphology.skeletonize_3d(img_3d)

    if squeeze_blobs:
        skel_3d = squeeze_skelton_blobs(skel_3d)

    if remove_surfaces:
        skel_3d = remove_skel_surfaces(skel_3d, surface_tol)

    return skel_3d


def squeeze_skelton_blobs(skel_3d):
    print(' - Filling voxels within closed surfaces in skeleton and re-thinning')
    print('\t\t WARNING: This may create parallel skeleton paths. Consider cleaning in network instead')
    skel_3d = skel_3d / np.amax(skel_3d)
    flat_skel = np.sum(skel_3d, axis=0)
    flat_skel = flat_skel > 0

    filled_skel = fill_lumen_ray_tracing(skel_3d, flat_skel, n_theta=3, n_phi=3, theta='sweep', mode='uniform',
                                         max_escape=1, path_l=1, skel=True)

    filled_skel = morphology.closing(filled_skel)
    filled_skel = morphology.dilation(filled_skel)
    re_skel = skeleton_3d((skel_3d + filled_skel) > 0)

    return re_skel


@njit(parallel=True)
def remove_skel_surfaces(img_skel, tol=5):
    print(' - Removing residual surfaces from skeletonization')
    print('\t\t WARNING: This may disconnect regions. Consider cleaning in network instead')
    img_skel = img_skel / np.amax(img_skel)
    dim = img_skel.shape
    skel_mask = np.ones(dim)
    dim = list(dim)

    for z in prange(1, dim[0] - 1):
        for x in prange(1, dim[1] - 1):
            for y in prange(1, dim[2] - 1):
                if img_skel[z, x, y] and np.sum(img_skel[z - 1:z + 2, x - 1:x + 2, y - 1:y + 2]) >= tol:
                    skel_mask[z, x, y] = 0

    return skel_mask * img_skel
