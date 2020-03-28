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


def pre_process_fill_lumen_ball(img_3d, img_mask):
    dims = img_3d.shape
    mask_dim = img_mask.shape
    if mask_dim[0] != dims[1] or mask_dim[1] != dims[2]:
        img_mask = transform.resize(img_mask, (dims[1], dims[2]))
        img_mask = filters.gaussian(img_mask, sigma=((dims[1]/mask_dim[0]) / 3.0, (dims[2]/mask_dim[1]) / 3.0))
        img_mask = img_mask > 0.5
        plt.imshow(img_mask)
        plt.show()

    return img_3d, img_mask

@njit(parallel=True)
def get_ball_filter(r):
    n = 2 * r - 1
    a, b = int(n / 2), int(n / 2)
    ball_filter_blank = np.zeros((1, n, n))
    ball_filter = ball_filter_blank.copy()

    for b_slice in range(0, r):
        mask = np.zeros((n, n))
        for x in prange(n):
            for y in prange(n):
                if (a - x) ** 2 + (b - y) ** 2 <= b_slice ** 2:
                    mask[x, y] = 1

        plane = ball_filter_blank
        plane[0] = mask
        ball_filter = np.append(ball_filter, plane, axis=0)

    ball_mask = ball_filter
    for b_slice in range(1, r):
        plane = ball_filter_blank
        plane[0] = ball_filter[r - b_slice]
        ball_mask = np.append(ball_mask, plane, axis=0)

    ball_mask = ball_mask[1::]

    return ball_mask



@njit(parallel=True)
def fill_lumen_ball(img_3d, img_mask,
                    maxr=15, minr=1,
                    thresh_pct=0.1, method='octants', thresh_num_oct=5, thresh_num_oct_op=4,
                    max_iters=1):

    r = maxr
    dims = img_3d.shape
    dim_z, dim_x, dim_y = dims[0], dims[1], dims[2]
    lumen_mask_tot = np.zeros(dims)

    print(' - Filling lumen using ball of radius:')
    while r >= minr:
        print(r)

        num_changes, its = 1, 0
        ball_mask = get_ball_filter(r)

        if method == 'octants' or method == 'pairs':
            oct_max_score = np.sum(ball_mask[0:r+1, 0:r+1, 0:r+1])
            thresh = thresh_pct*oct_max_score

        elif method == 'ball':
            ball_max_score = np.sum(ball_mask)
            thresh = thresh_pct*ball_max_score

        while num_changes > 0 and its < max_iters:
            its = its + 1

            lumen_mask = np.zeros(dims, np.int8)
            for z in prange(r, dim_z-r):
                for x in prange(r, dim_x-r):
                    for y in prange(r, dim_y-r):
                        if img_mask[x, y] and not img_3d[z, x, y]:
                            cube_region = img_3d[z-r+1:z+r, x-r+1:x+r, y-r+1:y+r]
                            filtered_cube = cube_region*ball_mask
                            total_hot = np.sum(filtered_cube)

                            if method == 'ball':
                                if total_hot >= thresh:
                                    lumen_mask[z, x, y] = 1

                            if method == 'octants' or method == 'pairs':
                                oct_count = 0
                                oct_op_pairs = 0

                                octants = np.zeros(8)
                                octants[0] = np.sum(filtered_cube[0:r+1, 0:r+1, 0:r+1])
                                octants[1] = np.sum(filtered_cube[r:2*r+1, r:2*r+1, r:2*r+1])

                                octants[2] = np.sum(filtered_cube[0:r+1, r:2*r+1, 0:r+1])
                                octants[3] = np.sum(filtered_cube[r:2*r+1, 0:r+1, r:2*r+1])

                                octants[4] = np.sum(filtered_cube[0:r + 1, 0:r + 1, r:2*r+1])
                                octants[5] = np.sum(filtered_cube[r:2*r+1, r:2*r+1, 0:r + 1])

                                octants[6] = np.sum(filtered_cube[r:2*r+1, 0:r + 1, 0:r + 1])
                                octants[7] = np.sum(filtered_cube[0:r + 1, r:2*r+1, r:2*r+1])


                                for oc in prange(8):
                                    oct_count += octants[oc] > thresh
                                    if oc % 2 == 0:
                                        oct_op_pairs += octants[oc] > thresh and octants[oc+1] > thresh

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



@njit()
def fill_lumen(img_3d, rl_xy=(5, 5), rh_xy=(10, 10), rz=1, thresh=10, iters=3):
    rxl = rl_xy[0]
    ryl = rl_xy[1]

    rxh = rh_xy[0]
    ryh = rh_xy[1]

    dims = img_3d.shape
    lumen_mask = np.zeros(dims, np.int64)
    for i in range(0, iters):
        for z in range(rz, dims[0]):
            for x in range(rxh, dims[1]):
                for y in range(ryh, dims[2]):
                    if img_3d[z, x, y] == 0:
                        nb_left = np.sum(img_3d[z - rz:z + rz + 1, x - rxh:x - rxl + 1, y - ryh:y + ryh + 1])
                        nb_right = np.sum(img_3d[z - rz:z + rz + 1, x + rxl:x + rxh + 1, y - ryh:y + ryh + 1])
                        nb_top = np.sum(img_3d[z - rz:z + rz + 1, x - rxh:x + rxh + 1, y + ryl:y + ryh + 1])
                        nb_bot = np.sum(img_3d[z - rz:z + rz + 1, x - rxh:x + rxh + 1, y - ryh:y - ryl + 1])

                        if (nb_left > thresh and nb_right > thresh) and (nb_top > thresh and nb_bot > thresh):
                            lumen_mask[z, x, y] = 1

        img_3d = img_3d + lumen_mask

    return lumen_mask


def show_lumen_fill(img_25d, l_fill):
    dims = img_25d.shape
    for p in range(0, dims[0]):
        print('Z-Slice %d of %d' % (p, dims[0] - 1))
        fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
        ax[0].set_title('Original')
        ax[0].imshow(img_25d[p], cmap='gray')

        ax[1].set_title('Lumen Mask')
        ax[1].imshow(l_fill[p], cmap='Blues')

        ax[2].set_title('Original + Lumen Mask')
        ax[2].imshow(img_25d[p], cmap='gray')
        ax[2].imshow(l_fill[p], cmap='Blues', alpha=0.5)

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
        img_out.append(st2.close_binary(thresh, k=1, plot=True, verbose=False))

    img_out = np.asarray(img_out)
    img_out = img_out/np.amax(img_out)
    img_out = morphology.binary_opening(img_out)
    img_out = morphology.binary_closing(img_out)
    img_out = st2.get_largest_connected_region(img_out, plot=False)
    return img_out


def skeleton_3d(img_3d):
    print(' - Thinning to 3D skeleton.')
    warnings.filterwarnings('ignore')
    skel_3d = morphology.skeletonize_3d(img_3d)
    return skel_3d
