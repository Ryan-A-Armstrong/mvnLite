import numpy as np

def smooth_dtransform(img_dist, kernel=(100, 100), step=(1, 1), shift_pct=0.1):
    print('\t - Transforming euclidean distance to radial distance.')
    print('\t\t - Kernel size: ' + str(kernel) + '. Step size: ' + str(step) + '.')
    dim = img_dist.shape
    x_k = kernel[0]
    y_k = kernel[1]
    x_step = step[0]
    y_step = step[1]

    img_out = np.copy(img_dist)
    for x in range(0, dim[0] - x_k + 1, x_step):
        for y in range(0, dim[1] - y_k + 1, y_step):
            k_array = img_dist[x:x+x_k, y:y+y_k]

            k_max = np.amax(k_array)
            k_min = np.amin(k_array)

            img_out[x:x+x_k, y:y+y_k] = np.sqrt((k_max - k_min)**2 - (k_max - k_array)**2) + k_min

    img_out = (img_out - np.amax(img_out)*shift_pct).clip(min=0)

    return img_out


def img_dist_to_img_volume(img_dist):
    print('\t - Projecting into 3space')
    dims = img_dist.shape

    maxr = int(np.ceil(np.amax(img_dist)))
    thickness = 2 * int(maxr) + 3
    zc = int(thickness / 2)

    img_3d = np.zeros((zc, dims[0], dims[1]))

    for z_slice in range(0, zc):
        img_3d[z_slice] = (zc - np.ceil(img_dist)) <= z_slice

    img_3d = np.append(img_3d, np.flip(img_3d, axis=0), axis=0)

    return img_3d



