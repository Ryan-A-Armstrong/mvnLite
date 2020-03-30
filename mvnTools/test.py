from mvnTools.Pipelines import std_2d_segment, std_3d_segment, segment_2d_to_meshes, generate_2d_network
from mvnTools.SegmentTools2d import skeleton, distance_transform
from mvnTools.SegmentTools3d import get_ellipsoid_filter
from skimage import io
from matplotlib import pyplot as plt
import timeit
'''
img_enhanced, img_mask, img_skel, img_dist, scaled = std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
                                                                    connected=True)
verts, faces, mesh = segment_2d_to_meshes(img_dist, img_skel, all_plots=True)
G = generate_2d_network(img_skel, img_dist,
                        near_node_tol=5, length_tol=1,
                        img_enhanced=img_enhanced, plot=True)
'''
img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/validateTJunction/400x50x50_20x100x10_hollow_4.tif', (1, 1),
                   connected=True, dila_gauss=0, all_plots=False)
#segment_2d_to_meshes(img_dist, img_skel, all_plots=True)
#generate_2d_network(img_skel, img_dist)
img_2d_stack = img_original.get_pages()
std_3d_segment(img_2d_stack, img_mask, (1, 1, 4),
               wth_k=0, bth_k=0, maxr=0, theta='xy',
               plot_lumen_fill=False, ellisoid_method='pairs',  thresh_num_oct_op=4, thresh_num_oct=8, n_phi=12, n_theta=12,
               plot_slices=False, window_size=(1, 1, 1))


#(1.2420, 1.2420, 1.14)
#(2.4859, 2.4859, 6.0000)