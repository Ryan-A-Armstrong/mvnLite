import os

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.external import tifffile as tif


def volume_density_data(img_binary, units=1, output_dir='', name='', connected=False, source='', save_ims=False):
    img_binary = np.asarray(img_binary, 'int')
    invert = -(img_binary - 1)
    non_zero_dim = np.nonzero(img_binary)
    minz, maxz = min(non_zero_dim[0]), max(non_zero_dim[0])
    minx, maxx = min(non_zero_dim[1]), max(non_zero_dim[1])
    miny, maxy = min(non_zero_dim[2]), max(non_zero_dim[2])

    max_volume = ((maxz - minz) * (maxx - minx) * (maxy - miny)) * units**3

    volume = np.sum(img_binary) * units ** 3
    vascular_density = float(volume / max_volume)

    dist_to_vessel = distance_transform_edt(invert[minz:maxz, minx:maxx, miny:maxy]) * units
    avg_dist_to_vessel = np.sum(dist_to_vessel) / (len(np.nonzero(dist_to_vessel)[0]))
    max_dist_to_vessel = np.amax(dist_to_vessel)

    plane_avg = []
    plane_max = []
    for i in dist_to_vessel:
        plane_avg.append(np.sum(i) / (len(np.nonzero(i)[0])))
        plane_max.append(np.amax(i))

    avg_plane_max = np.mean(plane_max)
    plane_avg = np.asarray(plane_avg, 'int')
    plane_max = np.asarray(plane_max, 'int')

    if not os.path.isdir(output_dir + 'masks3D/'):
        os.mkdir(output_dir + 'masks3D/')
    if not os.path.isdir(output_dir + 'masks3D/' + name + '/'):
        os.mkdir(output_dir + 'masks3D/' + name + '/')

    dir = output_dir + 'masks3D/' + name + '/'

    binary_data = open(dir + 'analysis.txt', 'a')
    binary_data.write('\n==============================================================================\n')
    binary_data.write('Binary image generated from %s.\nConnected is %r\n\n' % (source, connected))
    binary_data.write(
        '(%d um, %d um, %d um)\tBounding box dimensions (z, x, y)\n' % ((maxz - minz)*units,
                                                                        (maxx - minx)*units,
                                                                        (maxy - miny)*units))
    binary_data.write('%d um^3\tTotal volume\n' % volume)
    binary_data.write('%f\tVascular density (volume vessel/volume space)\n' % vascular_density)
    binary_data.write('%f um\tAverage distance from vessel wall\n' % avg_dist_to_vessel)
    binary_data.write('%f um\tMaximum distance from vessel wall\n' % max_dist_to_vessel)
    binary_data.write('%f um\tAverage maximum distance from vessel wall per z-plane\n' % avg_plane_max)
    binary_data.write('\nAverage distance to vessel wall per z-plane (rounded to um):\n' + str(plane_avg) + '\n\n')
    binary_data.write('\nMaximum distance to vessel wall per z-plane (rounded to um):\n' + str(plane_max) + '\n\n')
    binary_data.write('==============================================================================\n')
    binary_data.close()

    if save_ims:
        if connected:
            tif.imsave(
                output_dir + 'masks3D/' + name + '/' + source + '-connected-mask-' + ('-%dum-pix' % units) + '.tif',
                np.asarray(img_binary, 'uint8'), bigtiff=True)
            tif.imsave(
                output_dir + 'masks3D/' + name + '/' + source + '-connected-dist-' + ('-%dum-pix' % units) + '.tif',
                np.asarray(dist_to_vessel, 'uint8'), bigtiff=True)
        else:
            tif.imsave(
                output_dir + 'masks3D/' + name + '/' + source + '-unconnected-mask-' + ('-%dum-pix' % units) + '.tif',
                np.asarray(img_binary, 'uint8'), bigtiff=True)
            tif.imsave(
                output_dir + 'masks3D/' + name + '/' + source + '-unconnected-dist-' + ('-%dum-pix' % units) + '.tif',
                np.asarray(dist_to_vessel, 'uint8'), bigtiff=True)
