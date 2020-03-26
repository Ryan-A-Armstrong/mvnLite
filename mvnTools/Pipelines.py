from mvnTools.TifStack import TifStack as ts
from mvnTools.SegmentTools2d import SegmentTools2d as st2
from mvnTools.Display import Display as d

from skimage.morphology import dilation

contrasts = {'original': 0, 'rescale': 1, 'equalize': 2, 'adaptive': 3}


class Pipelines:
    def __init__(self, image_array):
        pass


def std_2d_segment(tif_file,
                   contrast_method='rescale',
                   thresh_method='entropy',
                   rwalk_thresh=(0, 0),
                   bth_k=3,
                   wth_k=3,
                   open_first=False,
                   open_k=0,
                   close_k=5,
                   all_plots=True,
                   review_plot=True):

    img_original = ts(tif_file,  page_list=False, flat=True)
    img_original_flat = img_original.get_flat()

    enhance_contrasts = st2.contrasts(img_original_flat, plot=all_plots, adpt=0.04)
    img_enhanced = enhance_contrasts[contrasts[contrast_method]]
    img = st2.black_top_hat(img_enhanced, plot=all_plots, k=bth_k)
    img = st2.white_top_hat(img, plot=all_plots, k=wth_k)
    dilated = st2.background_dilation(img, gauss=2, plot=all_plots)

    if thresh_method == 'random-walk':
        low = rwalk_thresh[0]
        high = rwalk_thresh[1]
        st2.random_walk_thresh(dilated, low, high, plot=all_plots)
    else:
        mask = st2.cross_entropy_thresh(dilated, plot=all_plots, verbose=False)

    if open_first:
        mask = st2.open_binary(mask, k=open_k, plot=all_plots)
        mask = st2.close_binary(mask, k=close_k, plot=all_plots)
    else:
        mask = st2.close_binary(mask, k=close_k, plot=all_plots)
        mask = st2.open_binary(mask, k=open_k, plot=all_plots)

    img_skel = st2.skeleton(mask, plot=all_plots)
    img_dist = st2.distance_transform(mask, plot=all_plots)

    if all_plots or review_plot:
        d.review_2d_results(img_enhanced, mask, dilation(img_skel), img_dist)

    return img_enhanced, mask, img_skel, img_dist
