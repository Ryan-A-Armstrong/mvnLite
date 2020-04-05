'''
img_enhanced, img_mask, img_skel, img_dist, scaled = std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
                                                                    connected=True)
verts, faces, mesh = segment_2d_to_meshes(img_dist, img_skel, all_plots=True)
G = generate_2d_network(img_skel, img_dist,
                        near_node_tol=5, length_tol=1,
                        img_enhanced=img_enhanced, plot=True)

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
'''

#driver = IO('/home/ryan/Desktop/mvn-analysis/inputs/original_sample.txt', echo_inputs=True)
#driver = IO('/home/ryan/Desktop/mvn-analysis/inputs/09092019 gel region 10x.txt', echo_inputs=True)

from mvnTools import Network2d as nw2
import numpy as np
from mvnTools.Pipelines import std_2d_segment
from matplotlib import pyplot as plt

img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
                   (1.5, 1.5), connected2D=False, generate_mesh_25=False,
                   output_dir='/home/ryan/Desktop/mvn-analysis/outputs/', all_plots=False, review_plot=False)
test = nw2.Network2d(img_skel, img_dist, near_node_tol=1, length_tol=1, plot=True, img_enhanced=img_enhanced)

img_dir = test.img_dir
leftdiag = img_dir == 1
vert = img_dir == 2
rightdiag = img_dir == 3
horizontal = img_dir == 4

plt.imshow(leftdiag)
plt.show()

plt.imshow(vert)
plt.show()

plt.imshow(rightdiag)
plt.show()

plt.imshow(horizontal)
plt.show()

plt.imshow(leftdiag + rightdiag + vert + horizontal)
plt.show()
