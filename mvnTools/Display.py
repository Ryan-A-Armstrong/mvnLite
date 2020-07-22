from matplotlib import pyplot as plt
# from mayavi import mlab
from skimage import exposure

'''
Code is taken from examples on skimage documentation. 
Need to return to properly cite
'''


def compare2(original, filtered, filter_name, cmap='gray'):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=cmap)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=cmap)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()


def compare3(im1, im2, im3, title1='Image 1', title2='Image 2', title3='Image 3', cmap='gray'):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
    ax[0].set_title(title1)
    ax[0].imshow(im1, cmap=cmap)
    ax[1].set_title(title2)
    ax[1].imshow(im2, cmap=cmap)
    ax[2].set_title(title3)
    ax[2].imshow(im3, cmap=cmap)

    plt.show()


def plot_img_and_hist(image, axes, bins=256):
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def review_2d_results(original, mask, dskeleton, distance, saving=False, units=1):
    dim = original.shape
    fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
    ax[0].set_title('Original')
    ax[0].imshow(original, cmap='gray')
    ax[0].hlines(dim[0] - 20, dim[1] - 100 / units - 20, dim[1] - 20, linewidth=4)
    ax[0].axis('off')

    ax[1].set_title('Original + Binary Mask')
    ax[1].imshow(original, cmap='gray')
    ax[1].imshow(mask, cmap='Blues', alpha=0.5)
    ax[1].hlines(dim[0] - 20, dim[1] - 100 / units - 20, dim[1] - 20, linewidth=4)
    ax[1].axis('off')

    ax[2].set_title('Original + (dilated) Skeleton + Distance Transform')
    ax[2].imshow(original, cmap='gray')
    ax[2].imshow(dskeleton, cmap='Blues', alpha=0.5)
    ax[2].imshow(distance, cmap='Reds', alpha=0.5)
    ax[2].hlines(dim[0] - 20, dim[1] - 100 / units - 20, dim[1] - 20, linewidth=4)
    ax[2].axis('off')

    plt.tight_layout()
    if saving:
        return fig

    plt.show()

'''
def visualize_dist(dist, offscreen=False):
    mlab.options.offscreen = offscreen
    mlab.surf(dist)
    mlab.show()


def visualize_mesh(mesh_data, offscreen=False, title=''):
    mlab.options.offscreen = offscreen
    verts = mesh_data[0]
    faces = mesh_data[1]
    if len(title) > 0:
        mlab.figure(title)
    mlab.triangular_mesh([vert[0] for vert in verts],
                         [vert[1] for vert in verts],
                         [vert[2] for vert in verts], faces)
    mlab.show()
'''