from mvnTools.Pipelines import std_2d_segment
import networkx as nx
from mvnTools.Network2d import Network2d
from matplotlib import pyplot as plt
from mvnTools.Network2dTools import get_directionality_pct_pix, get_directionality_pct_mask,\
    plot_directionality, closeness_vitatlity
'''
img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/Shared/09092019_96h_beads.tif', (1.243, 1.243),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test = Network2d(img_skel, img_dist)
#get_directionality_pct_pix(test.img_dir, img_dist, weighted=False)
#get_directionality_pct_pix(test.img_dir, img_dist, weighted=True)
#get_directionality_pct_mask(img_mask)
#plot_directionality(test.img_dir, img_enhanced)
closeness_vitatlity(test.G, img_enhanced)

img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/10x.tif', (2.4859, 2.4859),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test2 = Network2d(img_skel, img_dist)
#get_directionality_pct_pix(test2.img_dir, img_dist, weighted=False)
#get_directionality_pct_pix(test2.img_dir, img_dist, weighted=True)
#get_directionality_pct_mask(img_mask)
#plot_directionality(test2.img_dir, img_enhanced)
closeness_vitatlity(test2.G, img_enhanced)
'''
img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif', (1.242, 1.242),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test3 = Network2d(img_skel, img_dist, near_node_tol=5)
test4 = Network2d(img_skel, img_dist, near_node_tol=10)
test5 = Network2d(img_skel, img_dist, near_node_tol=15)
test6 = Network2d(img_skel, img_dist, near_node_tol=20)
test7 = Network2d(img_skel, img_dist, near_node_tol=20)
test8 = Network2d(img_skel, img_dist, near_node_tol=30)
test9 = Network2d(img_skel, img_dist, near_node_tol=40)
test10 = Network2d(img_skel, img_dist, near_node_tol=50)

plt.hist([test3.lengths, test4.lengths, test5.lengths, test6.lengths, test7.lengths, test8.lengths, test9.lengths, test10.lengths], bins=20, density=True)
plt.show()

#get_directionality_pct_pix(test2.img_dir, img_dist, weighted=False)
#get_directionality_pct_pix(test2.img_dir, img_dist, weighted=True)
#get_directionality_pct_mask(img_mask)
#plot_directionality(test2.img_dir, img_enhanced)
#closeness_vitatlity(test3.G, img_enhanced)


#plt.hist([test.lengths, test2.lengths, test3.lengths], bins=20, density=True)
#plt.show()

#plt.hist([test.volumes, test2.volumes, test3.volumes], bins=20, density=True)
#plt.show()

#plt.hist([test.volumes, test2.volumes, test3.volumes], bins=20, density=True)
#plt.show()

'''
plt.hist([test.contractions, test2.contractions, test3.contractions], bins=20, density=True)
plt.title('contractions')
plt.show()


print(test.fractal_scores)
print(test2.fractal_scores)
print(test3.fractal_scores)
plt.hist([test.fractal_scores, test2.fractal_scores, test3.fractal_scores], bins=20, density=True)
plt.show()
'''
#print('simularity')
#print(list(nx.algorithms.similarity.optimize_graph_edit_distance(test.G, test2.G)))