import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import exposure, img_as_float, img_as_ubyte, filters, measure
from skimage.filters import threshold_li
from skimage.filters.thresholding import _cross_entropy
from skimage.morphology import white_tophat, black_tophat, disk, reconstruction, opening, closing, dilation, skeletonize
from skimage.segmentation import random_walker

from mvnTools import Display as d

'''
Need to cite the code taken from skimage examples
'''


def white_top_hat(image, k=2, plot=False, verbose=True):
    if verbose:
        print('\t - Removing white peaks using disk of radius %d' % k)
    res = white_tophat(image, disk(k))
    if plot:
        d.compare3(image, res, image - res,
                   title1='Original', title2='White Top Hat', title3='Complimentary')

    return image - res


def black_top_hat(image, k=2, plot=False, verbose=True):
    if verbose:
        print('\t - Removing black holes using disk of radius %d' % k)
    res = black_tophat(image, disk(k))
    if plot:
        d.compare3(image, res, image + res,
                   title1='Original', title2='Black Top Hat', title3='Complimentary')

    return image + res


def contrasts(img, plot=True, adpt=0.04):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=adpt)

    if plot:
        # Display results
        fig = plt.figure(figsize=(15, 15))
        axes = np.zeros((2, 4), dtype=np.object)
        axes[0, 0] = fig.add_subplot(2, 4, 1)
        for i in range(1, 4):
            axes[0, i] = fig.add_subplot(2, 4, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
        for i in range(0, 4):
            axes[1, i] = fig.add_subplot(2, 4, 5 + i)

        ax_img, ax_hist, ax_cdf = d.plot_img_and_hist(img, axes[:, 0])
        ax_img.set_title('Original image')

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_img, ax_hist, ax_cdf = d.plot_img_and_hist(img_rescale, axes[:, 1])
        ax_img.set_title('Contrast stretching')

        ax_img, ax_hist, ax_cdf = d.plot_img_and_hist(img_eq, axes[:, 2])
        ax_img.set_title('Histogram equalization')

        ax_img, ax_hist, ax_cdf = d.plot_img_and_hist(img_adapteq, axes[:, 3])
        ax_img.set_title('Adaptive equalization')

        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(np.linspace(0, 1, 5))

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()

    return img, img_rescale, img_eq, img_adapteq


def background_dilation(image, gauss=1/3, plot=True, verbose=True):
    if verbose:
        print('\t - Extracting vessels from background and filtering with sigma=%0.4f' % gauss)
    if not gauss:
        return image

    image = img_as_float(image)
    image = gaussian_filter(image, gauss)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')

    if plot:
        d.compare2(image, dilated, 'Reconstruction via Dilation')

    return dilated


def cross_entropy_thresh(image, plot=True, plot_entropy=False, verbose=True):
    if verbose:
        print('\t - Thresholding by minimizing cross entropy')
    image = np.clip(image, -1.0, 1.0)
    im = img_as_ubyte(image)

    # thresholds = np.arange(np.min(im) + 1.5, np.max(im) - 1.5)
    # entropies = [_cross_entropy(im, t) for t in thresholds]
    #
    # optimal_im_threshold = thresholds[np.argmin(entropies)]
    #
    # if plot:
    #     d.compare2(im, im > optimal_im_threshold, 'Cross Entropy Threshold')
    #
    # if plot_entropy:
    #     plt.plot(thresholds, entropies)
    #     plt.xlabel('thresholds')
    #     plt.ylabel('cross-entropy')
    #     plt.vlines(optimal_im_threshold,
    #                ymin=np.min(entropies) - 0.05 * np.ptp(entropies),
    #                ymax=np.max(entropies) - 0.05 * np.ptp(entropies))
    #     plt.title('optimal threshold')
    #     plt.show()
    #
    #     print('The brute force optimal threshold is:', optimal_im_threshold)
    #     print('The computed optimal threshold is:', filters.threshold_li(im))
    #
    # return im > optimal_im_threshold

    return im > threshold_li(im)


def random_walk_thresh(image, low, high, plot=True, verbose=True):
    if verbose:
        print('\t - Thresholding by random walk algorithm with low=%0.4f and high=%0.4f' % (low, high))
    data = image
    print('\t\t - Image pixel range: %0.4f, %0.4f' % (np.amin(data), np.amax(data)))

    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < low] = 1
    markers[data > high] = 2

    # Run random walker algorithm
    labels = random_walker(data, markers, beta=1, mode='bf')

    if plot:
        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                            sharex=True, sharey=True)
        ax1.imshow(data, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Adaptize Equalization')
        ax2.imshow(markers, cmap='magma')
        ax2.axis('off')
        ax2.set_title('Markers')
        ax3.imshow(labels, cmap='gray')
        ax3.axis('off')
        ax3.set_title('Segmentation')

        fig.tight_layout()
        plt.show()
    return labels - 1


def open_binary(img_binary, k=5, plot=True, verbose=True):
    if verbose:
        print('\t - Removing artifacts with disk of radius %d' % k)
    opened = opening(img_binary, disk(k))
    if plot:
        d.compare2(img_binary, opened, 'Opening: k = %d' % k)
    return opened


def close_binary(img_binary, k=5, plot=True, verbose=True):
    if verbose:
        print('\t - Closing holes with disk of radius %d' % k)
    opened = closing(img_binary, disk(k))
    if plot:
        d.compare2(img_binary, opened, 'Closing: k = %d' % k)
    return opened


def skeleton(img_binary, plot=True, verbose=True):
    if verbose:
        print('\t - Thinning to skeleton')
    img_skel = skeletonize(img_binary, method='lee')

    if plot:
        d.compare2(img_binary, dilation(img_skel), 'Skeletonization (image dilated for viewing)')

    img_skel = img_skel/np.amax(img_skel)

    return img_skel


def distance_transform(img_binary, plot=True, verbose=True):
    if verbose:
        print('\t - Calculating euclidean distance transform')
    img_dist = ndi.distance_transform_edt(img_binary)

    if plot:
        d.compare2(img_binary, img_dist, 'Euclidean Distance Transform')

    return img_dist


def get_largest_connected_region(img_binary, plot=True, verbose=True):
    if verbose:
        print(' - Retaining only the largest connected region')
    img_lab = measure.label(img_binary, connectivity=2)
    regions = measure.regionprops(img_lab)
    val_max = 0
    label_max = 0
    for props in regions:
        if props.area > val_max:
            val_max = props.area
            label_max = props.label

    img_largest_binary = img_lab == label_max

    if plot:
        d.compare2(img_lab, img_largest_binary, 'Largest Connected Region', cmap='magma')

    return img_largest_binary
