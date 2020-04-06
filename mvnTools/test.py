from mvnTools.Pipelines import std_2d_segment
from mvnTools.Network2d import Network2d
from matplotlib import pyplot as plt
from mvnTools.Network2dTools import get_directionality_pct_pix, get_directionality_pct_mask,\
    plot_directionality, closeness_vitatlity

img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/OneDrive_2020-03-30/ForRyan/09092019_96h_ beads.tif', (1.2430, 1.2430),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test = Network2d(img_skel, img_dist)
get_directionality_pct_pix(test.img_dir, img_dist, weighted=False)
get_directionality_pct_pix(test.img_dir, img_dist, weighted=True)
get_directionality_pct_mask(img_mask)
plot_directionality(test.img_dir, img_enhanced)
closeness_vitatlity(test.G, img_enhanced)

img_enhanced, img_mask, img_skel, img_dist, img_original = \
    std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/10x.tif', (2.4859, 2.4859),
                   generate_mesh_25=False, review_plot=True, all_plots=False, connected2D=True)

test2 = Network2d(img_skel, img_dist)
get_directionality_pct_pix(test2.img_dir, img_dist, weighted=False)
get_directionality_pct_pix(test2.img_dir, img_dist, weighted=True)
get_directionality_pct_mask(img_mask)
plot_directionality(test.img_dir, img_enhanced)
closeness_vitatlity(test2.G, img_enhanced)

print(test.total_length)
print(test2.total_length)

plt.hist(test.lengths, bins=50, color='red', alpha=0.5, density=True)
plt.hist(test2.lengths, bins=50, color='blue', alpha=0.5, density=True)
plt.show()

plt.hist(test.volumes, bins=50, color='red', alpha=0.5, density=True)
plt.hist(test2.volumes, bins=50, color='blue', alpha=0.5, density=True)
plt.show()

plt.hist(test.surfaces, bins=50, color='red', alpha=0.5, density=True)
plt.hist(test2.surfaces, bins=50, color='blue', alpha=0.5, density=True)
plt.show()