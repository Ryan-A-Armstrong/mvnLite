from matplotlib import pyplot as plt
from skimage import img_as_float, exposure

'''
Code is taken from examples on skimage documentation. 
Need to return to properly cite
'''


class Display:

    def __init__(self):
        pass

    @staticmethod
    def compare2(original, filtered, filter_name):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')
        plt.show()

    @staticmethod
    def compare3(im1, im2, im3, title1='Image 1', title2='Image 2', title3='Image 3'):
        fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
        ax[0].set_title(title1)
        ax[0].imshow(im1, cmap='gray')
        ax[1].set_title(title2)
        ax[1].imshow(im2, cmap='gray')
        ax[2].set_title(title3)
        ax[2].imshow(im3, cmap='gray')

        plt.show()

    @staticmethod
    def plot_img_and_hist(image, axes, bins=256):
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    @staticmethod
    def review_2d_results(original, mask, dskeleton, distance):
        fig, ax = plt.subplots(ncols=3, figsize=(15, 8))
        ax[0].set_title('Original')
        ax[0].imshow(original, cmap='gray')

        ax[1].set_title('Original + Binary Mask + (dilated) Skeleton')
        ax[1].imshow(original, cmap='gray')
        ax[1].imshow(mask, cmap='Blues', alpha=0.5)

        ax[2].set_title('Original + (dilated) Skeleton + Distance Transform')
        ax[2].imshow(original, cmap='gray')
        ax[2].imshow(dskeleton, cmap='Blues', alpha=0.5)
        ax[2].imshow(distance, cmap='Reds', alpha=0.5)

        plt.show()
