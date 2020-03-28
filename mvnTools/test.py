from mvnTools.Pipelines import std_2d_segment, std_3d_segment, segment_2d_to_meshes, generate_2d_network
from mvnTools.SegmentTools2d import skeleton, distance_transform
from mvnTools.SegmentTools3d import get_ellipsoid_filter
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
