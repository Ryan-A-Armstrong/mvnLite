
class Outputs:
    # Tif outputs
    img_flat = None
    tif_pages = None

    # 2d outputs
    img_enhanced = None
    full_mask = None
    connected_mask = None
    euclid_dist_trans = None
    rounded_dist_trans = None

    # 3d outputs
    mesh_2d = None
    mesh_3d = None
    skel_3d = None

    # Graph outputs
    G = None

    def __init__(self):
        pass
