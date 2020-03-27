import numpy as np


def smooth_dtransform_auto(img_dist, img_skel):
    print('\t - Transforming euclidean distance to radial distance')
    print('\t\t - Automating kernel size along skeleton path')
    dim = img_dist.shape

    img_skel = img_skel/np.amax(img_skel)
    kernel_map = img_dist*img_skel

    img_round = np.zeros(img_dist.shape)
    count_map = np.zeros(img_dist.shape)

    for x in range(0, dim[0]):
        for y in range(0, dim[1]):
            if kernel_map[x, y] > 0:
                kr = np.ceil(kernel_map[x, y])
                x_min, y_min = int(x-kr-1), int(y-kr-1)
                x_max, y_max = int(x+kr+1), int(y+kr+1)

                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                if x_max >= dim[0]:
                    x_max = dim[0]
                if y_max >= dim[1]:
                    y_max = dim[1]

                k_array = img_dist[x_min:x_max, y_min:y_max]

                k_max = np.amax(k_array)
                k_min = np.amin(k_array)

                img_round[x_min:x_max, y_min:y_max] += np.sqrt((k_max - k_min)**2 - (k_max - k_array)**2) + k_min
                count_map[x_min:x_max, y_min:y_max] += 1

    print('\t\t - Smoothing transforms between kernels')
    count_map = np.where(count_map == 0, 1, count_map)
    img_round = img_round/count_map

    return img_round


def img_dist_to_img_volume(img_dist):
    print('\t - Projecting into z-dimension')
    dims = img_dist.shape

    maxr = int(np.ceil(np.amax(img_dist)))
    thickness = 2 * int(maxr) + 3
    zc = int(thickness / 2)

    img_3d = np.zeros((zc, dims[0], dims[1]))

    for z_slice in range(0, zc):
        img_3d[z_slice] = (zc - np.ceil(img_dist)) <= z_slice

    img_3d = np.append(img_3d, np.flip(img_3d, axis=0), axis=0)

    return img_3d



