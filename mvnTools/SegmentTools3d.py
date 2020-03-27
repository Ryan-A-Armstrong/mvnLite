import warnings

import mvnTools.SegmentTools2d as st2
import numpy as np
from numba import njit
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_li, threshold_sauvola


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


@njit()
def fill_lumen(img_3d, rl_xy, rh_xy, rz, thresh, iters=1):
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
