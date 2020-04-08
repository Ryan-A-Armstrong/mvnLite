from mvnTools.Pipelines import std_2d_segment
import networkx as nx
import pickle
from scipy import stats
import numpy as np
from mvnTools.Network2d import Network2d
from matplotlib import pyplot as plt
from mvnTools.Network2dTools import get_directionality_pct_pix, get_directionality_pct_mask,\
    plot_directionality, closeness_vitatlity

img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif', (1.242, 1.242),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test = Network2d(img_skel, img_dist, img_enhanced=img_enhanced, output_dir='/home/ryan/Desktop/mvn-analysis/outputs/',
                 save_files=True, plot=False, name='test')
'''
with open('test.graphclass', 'wb') as testfile:
    pickle.dump(test, testfile)
with open('test.graphclass', 'rb') as testfile:
    H = pickle.load(testfile)

print(H.lengths)
'''