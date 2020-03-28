from mvnTools.Pipelines import std_2d_segment, std_3d_segment, segment_2d_to_meshes, generate_2d_network
from mvnTools.SegmentTools2d import skeleton, distance_transform
from skimage import io
from matplotlib import pyplot as plt
'''
img_enhanced, img_mask, img_skel, img_dist, scaled = std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif')
verts, faces, mesh = segment_2d_to_meshes(img_dist, img_skel, all_plots=True)
G = generate_2d_network(img_skel, img_dist,
                        near_node_tol=5, length_tol=1,
                        img_enhanced=img_enhanced, plot=True, )
'''
std_3d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif')

import numpy as np
from numba import njit, prange

A = [1,2,3,4,5]
A = np.asarray(A)

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
    return (ball_mask)

print(prange_test(3))