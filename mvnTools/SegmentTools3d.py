import warnings

import mvnTools.SegmentTools2d as st2
import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from skimage import morphology, transform, filters
from skimage.morphology.selem import disk
from skimage.filters import threshold_li, threshold_sauvola
from matplotlib import pyplot as plt


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
    print('Interpolating and scaling z planes.')
    off_int = z_scale - int(z_scale)
    off_int_cum = 0

    img_3d = []
    for i in range(0, len(im_25d) - 1):
        top = im_25d[i + 1]
        bottom = im_25d[i]

        img_3d.append(bottom)
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
    n = 2 * r - 1
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


# function get_new_pos()
# input:   old_pos    -previous position of the photon
#          k          -trajectory (theta, phi)
#          path_l     -ray step size
#
# output:  new_pos   -new position(x,y,z)
@njit(fastmath=True)
def get_new_pos(old_pos, k,  path_l):
    #print('old pos')
    #print(old_pos)
    #print('k')
    #print(k)
    delz = path_l * np.cos(k[0])
    delx = path_l * np.sin(k[0]) * np.cos(k[1])
    dely = path_l * np.sin(k[0]) * np.sin(k[1])

    new_pos = (old_pos[0] + delz, old_pos[1] + delx, old_pos[2] + dely)
    #print('new pos')
    #print(new_pos)
    #print()
    return new_pos


@njit(parallel=True)
def fill_lumen_ray_tracing(img_3d, img_mask,
                           n_theta=3, n_phi=3, mode='uniform', max_escape=1,
                           path_l=1):
    print('Ray Tracing')
    dims = img_3d.shape
    lumen_mask = np.zeros(dims, np.int8)

    dims = list(dims)
    dim_z, dim_x, dim_y = dims[0], dims[1], dims[2]

    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi + 1)[1::]

    for z in prange(0, dim_z):
        for x in prange(0, dim_x):
            for y in prange(0, dim_y):
                if img_mask[x, y] and not img_3d[z, x, y]:
                    escaped = 0
                    free_path = escaped >= max_escape

                    for i_t in prange(n_theta):
                        if free_path:
                            break
                        if mode == 'random':
                            t = np.random.rand() * np.pi - np.pi / 2
                        else:
                            t = thetas[i_t]

                        for i_p in prange(n_phi):
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
                                pos = get_new_pos(pos, k,  path_l)
                                z_im = int(pos[0] + 0.5)
                                x_im = int(pos[1] + 0.5)
                                y_im = int(pos[2] + 0.5)

                                in_img = (0 <= z_im < dims[0]) and \
                                         (0 <= x_im < dims[1]) and \
                                         (0 <= y_im < dims[2])

                            if not in_img: # Loop broke because ray got to image edge
                                escaped = escaped + 1
                                free_path = True

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

    print('Filling lumen using ellipsoid.')
    print('Filter radius (xy)')
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
        ax[2].imshow(l_fill[p], cmap=cmap1, alpha=0.5*alpha)
        if l2_fill is not None:
            ax[2].imshow(l2_fill[p], cmap=cmap2, alpha=0.5*alpha)

        plt.show()


def img_2d_stack_to_binary_array(img_2d_stack, smooth=1):
    img_2d_stack = np.asarray(img_2d_stack)

    for im_plane in range(0, len(img_2d_stack)):
        im = img_2d_stack[im_plane]
        im = st2.black_top_hat(im, plot=False, k=3, verbose=False)
        im = st2.white_top_hat(im, plot=False, k=3, verbose=False)
        img_2d_stack[im_plane] = im

    img_out = []
    thresh_3d = threshold_sauvola(img_2d_stack, window_size=(7, 15, 15), k=0.3)

    for plane in range(0, len(thresh_3d)):
        thresh = thresh_3d[plane] > threshold_li(thresh_3d[plane])
        img_out.append(st2.close_binary(thresh, k=1, plot=False, verbose=False))

    img_out = np.asarray(img_out)
    img_out = img_out / np.amax(img_out)
    img_out = morphology.binary_opening(img_out)
    img_out = morphology.binary_closing(img_out)
    img_out = st2.get_largest_connected_region(img_out, plot=False)
    return img_out


def skeleton_3d(img_3d):
    print(' - Thinning to 3D skeleton.')
    warnings.filterwarnings('ignore')
    skel_3d = morphology.skeletonize_3d(img_3d)
    return skel_3d
