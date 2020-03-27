from mvnTools.Pipelines import std_2d_segment, segment_2d_to_meshes, generate_2d_network
from mvnTools.SegmentTools2d import skeleton, distance_transform
from skimage import io
from matplotlib import pyplot as plt

img_enhanced, img_mask, img_skel, img_dist, scaled = std_2d_segment(
    '/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
    connected=True, close_k=5, bth_k=3, wth_k=1, scale_xy=(1.2420, 1.2420), smooth=0)

G = generate_2d_network(img_skel, img_dist,
                        near_node_tol=25, length_tol=1,
                        img_enhanced=img_enhanced, plot=True, )
