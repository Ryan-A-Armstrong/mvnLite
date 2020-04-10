import ntpath
import os
import pickle

import numpy as np

from mvnTools.Network2d import Network2d
from mvnTools.Network2dTools import network_histograms
from mvnTools.Pipelines import std_2d_segment, std_3d_segment, network_2d_analysis


class IO:
    name = None
    input_dic = {
        # Input image and scale values (um / pixel)
        'TIF_FILE': None,
        'SCALE_X': None,
        'SCALE_Y': None,
        'SCALE_Z': None,

        # Piplelines to Run
        'SEGMENT_2D': False,
        'MESH_25D': False,
        'SEGMENT_3D': False,
        'MESH_3D': False,
        'SKEL_3D': False,
        'VOLUME_ANALYSIS': False,
        'NETWORK_2D_GEN': False,
        'NETWORK_2D_COMPARE': False,
        'NETWORK_3D_GEN': False,
        'NETWORK_3D_COMPARE=': False,

        # Save Parameters
        'OUTPUT_DIR': None,

        'SAVE_2D_MASK': True,
        'SAVE_2D_SKEL': True,
        'SAVE_2D_DIST': True,
        'SAVE_2D_DISPLAY': True,
        'SAVE_2D_REVIEW': True,

        'SAVE_25D_MESH': True,
        'SAVE_25D_MASK': False,
        'GENERATE_25D_VOLUME': False,

        'SAVE_3D_MASK': False,
        'SAVE_3D_SKEL': False,

        'SAVE_3D_MESH': True,
        'GENERATE_3D_VOLUME': False,
        'SAVE_3D_MESH_ROUND': True,
        'GENERATE_3D_ROUND_VOLUME': False,

        'SAVE_2D_NETWORK': True,
        'SAVE_3D_NETWORK': True,

        # Display Parameters
        'PLOT_ALL_2D': False,
        'REVIEW_PLOT_2D': False,

        'PLOT_25D_MESH': True,

        'PLOT_3D_THRESH_SLICES': False,
        'PLOT_LUMEN_FILL': False,

        'PLOT_3D_MESHES': True,
        'PLOT_3D_SKELS': True,

        'PLOT_NETWORK_GEN': True,
        'PLOT_NETWORK_DATA': False,

        # 2D Analysis
        'CONTRAST_METHOD': 'rescale',
        'THRESH_METHOD': 'entropy',
        'RWALK_THRESH_LOW': 0.75,
        'RWALK_THRESH_HIGH': 0.8,
        'BTH_K_2D': 3,
        'WTH_K_2D': 3,
        'DILA_GAUSS': 0.333,
        'OPEN_FIRST': False,
        'OPEN_K_2D': 0,
        'CLOSE_K_2D': 5,
        'CONNECTED_2D': False,
        'SMOOTH': 1.0,

        # 25D Analysis
        'H_PCT_25D:': 1,
        'CONNECTED_25D_MESH': True,
        'CONNECTED_25D_VOLUME': False,

        # 3D Analysis

        # Output parameters
        'CONNECTED_3D_MASK': False,
        'CONNECTED_3D_MESH': True,
        'CONNECTED_3D_SKEL': False,

        'SLICE_CONTRAST': 'original',
        'PRE_THRESH': 'sauvola',
        'WINDOW_SIZE_X': 15,
        'WINDOW_SIZE_Y': 15,
        'WINDOW_SIZE_Z': 7,

        'BTH_K_3D': 3,
        'WTH_K_3D': 3,
        'CLOSE_K_3D': 5,

        'ELLIPSOID_METHOD': 'octants',
        'MAXR': 15,
        'MINR': 1,
        'H_PCT_R': 0.5,
        'THRESH_PCT': 0.15,
        'THRESH_NUM_OCT': 5,
        'THRESH_NUM_OCT_OP': 3,
        'MAX_ITERS': 1,

        'RAY_TRACE_MODE': 'uniform',
        'THETA': 'exclude_z_pole',
        'N_THETA': 6,
        'N_PHI': 6,
        'MAX_ESCAPE': 2,
        'PATH_L': 1,

        'FILL_LUMEN_MESHING': True,
        'FILL_LUMEN_MESHING_MAX_ITS': 3,

        'ENFORCE_ELLIPSOID_LUMEN': True,
        'H_PCT_ELLIPSOID': 1.00,

        # Skeletonizaiton Parameters
        'SQUEEZE_SKEL_BLOBS': True,
        'REMOVE_SKEL_SURF': True,
        'SKEL_SURF_TOL': 5,
        'SKEL_CLOSING': True,

        # Graph Analysis
        'CONNECTED_NETWORK': True,
        'MIN_NODE_COUNT': 3,
        'NEAR_NODE_TOL': 5,
        'LENGTH_TOL': 1,
    }

    input_dic_setter = {
        # Input image and scale values (um / pixel)
        'TIF_FILE=': str,
        'SCALE_X=': float,
        'SCALE_Y=': float,
        'SCALE_Z=': float,

        # Piplelines to Run
        'SEGMENT_2D=': bool,
        'MESH_25D=': bool,
        'SEGMENT_3D=': bool,
        'MESH_3D=': bool,
        'SKEL_3D=': bool,
        'VOLUME_ANALYSIS=': bool,
        'NETWORK_2D_GEN=': bool,
        'NETWORK_2D_COMPARE=': bool,
        'NETWORK_3D_GEN=': bool,
        'NETWORK_3D_COMPARE=': bool,

        # Save Parameters
        'OUTPUT_DIR=': str,

        'SAVE_2D_MASK=': bool,
        'SAVE_2D_SKEL=': bool,
        'SAVE_2D_DIST=': bool,
        'SAVE_2D_DISPLAY=': bool,
        'SAVE_2D_REVIEW=': bool,

        'SAVE_25D_MESH=': bool,
        'SAVE_25D_MASK=': bool,
        'GENERATE_25D_VOLUME=': bool,

        'SAVE_3D_MASK=': bool,
        'SAVE_3D_SKEL=': bool,

        'SAVE_3D_MESH=': bool,
        'GENERATE_3D_VOLUME=': bool,
        'SAVE_3D_MESH_ROUND=': bool,
        'GENERATE_3D_ROUND_VOLUME=': bool,

        'SAVE_2D_NETWORK=': bool,
        'SAVE_3D_NETWORK=': bool,

        # Display Parameters
        'PLOT_ALL_2D=': bool,
        'REVIEW_PLOT_2D=': bool,

        'PLOT_25D_MESH=': bool,

        'PLOT_3D_THRESH_SLICES=': bool,
        'PLOT_LUMEN_FILL=': bool,

        'PLOT_3D_MESHES=': bool,
        'PLOT_3D_SKELS=': bool,

        'PLOT_NETWORK_GEN=': bool,
        'PLOT_NETWORK_DATA=': bool,

        # 2D Analysis
        'CONTRAST_METHOD=': str,
        'THRESH_METHOD=': str,
        'RWALK_THRESH_LOW=': float,
        'RWALK_THRESH_HIGH=': float,
        'BTH_K_2D=': int,
        'WTH_K_2D=': int,
        'DILA_GAUSS=': float,
        'OPEN_FIRST=': bool,
        'OPEN_K_2D=': int,
        'CLOSE_K_2D=': int,
        'CONNECTED_2D=': bool,
        'SMOOTH=': float,

        # 25D Analysis
        'H_PCT_25D=': float,
        'CONNECTED_25D_MESH=': bool,
        'CONNECTED_25D_VOLUME=': bool,

        # 3D Analysis

        # Output parameters
        'CONNECTED_3D_MASK=': bool,
        'CONNECTED_3D_MESH=': bool,
        'CONNECTED_3D_SKEL=': bool,

        # Thresholding parameters
        'SLICE_CONTRAST=': str,
        'PRE_THRESH=': str,
        'WINDOW_SIZE_X=': int,
        'WINDOW_SIZE_Y=': int,
        'WINDOW_SIZE_Z=': int,

        'BTH_K_3D=': int,
        'WTH_K_3D=': int,
        'CLOSE_K_3D=': int,

        'ELLIPSOID_METHOD=': str,
        'MAXR=': int,
        'MINR=': int,
        'H_PCT_R=': float,
        'THRESH_PCT=': float,
        'THRESH_NUM_OCT=': int,
        'THRESH_NUM_OCT_OP=': int,
        'MAX_ITERS=': int,

        'RAY_TRACE_MODE=': str,
        'THETA=': str,
        'N_THETA=': int,
        'N_PHI=': int,
        'MAX_ESCAPE=': int,
        'PATH_L=': int,

        'FILL_LUMEN_MESHING=': bool,
        'FILL_LUMEN_MESHING_MAX_ITS=': int,

        'ENFORCE_ELLIPSOID_LUMEN=': bool,
        'H_PCT_ELLIPSOID=': float,

        # Skeletonizaiton Parameters
        'SQUEEZE_SKEL_BLOBS=': bool,
        'REMOVE_SKEL_SURF=': bool,
        'SKEL_SURF_TOL=': int,
        'SKEL_CLOSING=': bool,

        # Graph Analysis
        'CONNECTED_NETWORK=': bool,
        'MIN_NODE_COUNT=': int,
        'NEAR_NODE_TOL=': int,
        'LENGTH_TOL=': int
    }

    def __init__(self, input_file, echo_inputs=False, silent_mode=False, all_plots_mode=False):
        with open(input_file, 'r') as f:
            inputs = f.read()

            for param in self.input_dic_setter:
                index_param = inputs.index(param)
                index_end_line = inputs.index('\n', index_param)
                val = inputs[index_param + len(param):index_end_line]

                if self.input_dic_setter[param] is int:
                    val = int(val)

                elif self.input_dic_setter[param] is float:
                    val = float(val)

                elif self.input_dic_setter[param] is bool:
                    val = bool(int(val))

                self.input_dic[param[:-1]] = val

        self.name = ntpath.basename(self.input_dic['TIF_FILE']).split('.')[0]
        if silent_mode:
            self.input_dic['PLOT_ALL_2D'] = False
            self.input_dic['REVIEW_PLOT_2D'] = False
            self.input_dic['PLOT_3D_THRESH_SLICES'] = False
            self.input_dic['PLOT_LUMEN_FILL'] = False
            self.input_dic['PLOT_3D_MESHES'] = False
            self.input_dic['PLOT_3D_SKELS'] = False
            self.input_dic['PLOT_NETWORK_GEN'] = False
            self.input_dic['PLOT_NETWORK_DATA'] = False

        if all_plots_mode:
            self.input_dic['PLOT_ALL_2D'] = True
            self.input_dic['REVIEW_PLOT_2D'] = True
            self.input_dic['PLOT_3D_THRESH_SLICES'] = True
            self.input_dic['PLOT_LUMEN_FILL'] = True
            self.input_dic['PLOT_3D_MESHES'] = True
            self.input_dic['PLOT_3D_SKELS'] = True
            self.input_dic['PLOT_NETWORK_GEN'] = True
            self.input_dic['PLOT_NETWORK_DATA'] = True

        if echo_inputs:
            for ins in self.input_dic:
                print(ins + '=' + str(self.input_dic[ins]))

        if self.input_dic['SEGMENT_2D'] or self.input_dic['MESH_25D']:
            std_2d_segment(self.input_dic['TIF_FILE'],
                           scale_xy=(self.input_dic['SCALE_X'], self.input_dic['SCALE_Y']),
                           scale_z=self.input_dic['SCALE_Z'],
                           h_pct=self.input_dic['H_PCT_25D'],
                           contrast_method=self.input_dic['CONTRAST_METHOD'],
                           thresh_method=self.input_dic['THRESH_METHOD'],
                           rwalk_thresh=(self.input_dic['RWALK_THRESH_LOW'], self.input_dic['RWALK_THRESH_HIGH']),
                           bth_k=self.input_dic['BTH_K_2D'],
                           wth_k=self.input_dic['WTH_K_2D'],
                           dila_gauss=self.input_dic['DILA_GAUSS'],
                           open_first=self.input_dic['OPEN_FIRST'],
                           open_k=self.input_dic['OPEN_K_2D'],
                           close_k=self.input_dic['CLOSE_K_2D'],
                           connected2D=self.input_dic['CONNECTED_2D'],
                           smooth=self.input_dic['SMOOTH'],
                           all_plots=self.input_dic['PLOT_ALL_2D'] and self.input_dic['SEGMENT_2D'],
                           review_plot=self.input_dic['REVIEW_PLOT_2D'] and self.input_dic['SEGMENT_2D'],
                           output_dir=self.input_dic['OUTPUT_DIR'],
                           name=self.name,
                           save_mask=self.input_dic['SAVE_2D_MASK'] and self.input_dic['SEGMENT_2D'],
                           save_skel=self.input_dic['SAVE_2D_SKEL'] and self.input_dic['SEGMENT_2D'],
                           save_dist=self.input_dic['SAVE_2D_DIST'] and self.input_dic['SEGMENT_2D'],
                           save_disp=self.input_dic['SAVE_2D_DISPLAY'] and self.input_dic['SEGMENT_2D'],
                           save_review=self.input_dic['SAVE_2D_REVIEW'] and self.input_dic['SEGMENT_2D'],
                           generate_mesh_25=self.input_dic['MESH_25D'],
                           connected_mesh=self.input_dic['CONNECTED_25D_MESH'],
                           connected_vol=self.input_dic['CONNECTED_25D_VOLUME'],
                           plot_25d=self.input_dic['PLOT_25D_MESH'],
                           save_volume_mask=self.input_dic['SAVE_25D_MASK'] or self.input_dic['VOLUME_ANALYSIS'],
                           save_surface_meshes=self.input_dic['SAVE_25D_MESH'],
                           generate_volume_meshes=self.input_dic['GENERATE_25D_VOLUME'])

        if self.input_dic['NETWORK_2D_GEN']:
            img_enhanced, img_mask, img_skel, img_dist, img_original = \
                std_2d_segment(self.input_dic['TIF_FILE'],
                               scale_xy=(self.input_dic['SCALE_X'], self.input_dic['SCALE_Y']),
                               contrast_method=self.input_dic['CONTRAST_METHOD'],
                               thresh_method=self.input_dic['THRESH_METHOD'],
                               rwalk_thresh=(self.input_dic['RWALK_THRESH_LOW'], self.input_dic['RWALK_THRESH_HIGH']),
                               bth_k=self.input_dic['BTH_K_2D'],
                               wth_k=self.input_dic['WTH_K_2D'],
                               dila_gauss=self.input_dic['DILA_GAUSS'],
                               open_first=self.input_dic['OPEN_FIRST'],
                               open_k=self.input_dic['OPEN_K_2D'],
                               close_k=self.input_dic['CLOSE_K_2D'],
                               connected2D=self.input_dic['CONNECTED_NETWORK'],
                               smooth=self.input_dic['SMOOTH'],
                               all_plots=False,
                               review_plot=False,
                               output_dir=self.input_dic['OUTPUT_DIR'],
                               name=self.name,
                               save_mask=False,
                               save_skel=False,
                               save_dist=False,
                               save_disp=False,
                               save_review=False,
                               generate_mesh_25=False,
                               connected_mesh=False,
                               connected_vol=False,
                               plot_25d=False,
                               save_volume_mask=False,
                               save_surface_meshes=False,
                               generate_volume_meshes=False)

            units = img_original.units
            network_object = Network2d(img_skel,
                                       img_dist,
                                       units=units,
                                       near_node_tol=self.input_dic['NEAR_NODE_TOL'],
                                       length_tol=self.input_dic['LENGTH_TOL'],
                                       min_nodes=self.input_dic['MIN_NODE_COUNT'],
                                       plot=self.input_dic['PLOT_NETWORK_GEN'],
                                       img_enhanced=img_enhanced,
                                       output_dir=self.input_dic['OUTPUT_DIR'],
                                       name=self.name,
                                       save_files=self.input_dic['SAVE_2D_NETWORK'])

            if self.input_dic['SAVE_2D_NETWORK']:
                if not os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks/'):
                    os.mkdir(self.input_dic['OUTPUT_DIR'] + 'networks/')
                if not os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/'):
                    os.mkdir(self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/')
                save_path = self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/'

                save_obj = save_path + 'network_object.gpickle'
                with open(save_obj, 'wb') as network:
                    pickle.dump(network_object, network)

            network_2d_analysis(network_object, output_dir=save_path, name=self.name)

        if self.input_dic['NETWORK_2D_COMPARE']:
            graph_objs = []
            names = []
            if os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks/'):
                dirs = [x[0] for x in os.walk(self.input_dic['OUTPUT_DIR'] + 'networks/')]
                for graphs in dirs:
                    if os.path.isfile(graphs + '/network_object.gpickle'):
                        with open(graphs + '/network_object.gpickle', 'rb') as graph_obj_file:
                            G = pickle.load(graph_obj_file)
                            graph_objs.append(G)
                            names.append(os.path.basename(graphs))
                            print(os.path.basename(graphs))

            if not os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks-compare/'):
                os.mkdir(self.input_dic['OUTPUT_DIR'] + 'networks-compare/')

            lengths, volumes, surfaces, radii, contractions, fractal_scores = [], [], [], [], [], []
            print('Comparing %d networks' % len(graph_objs))
            for g in graph_objs:
                lengths.append(g.lengths)
                volumes.append(g.volumes)
                surfaces.append(g.surfaces)
                radii.append(g.radii)
                contractions.append(g.contractions)
                fractal_scores.append(g.fractal_scores)

            network_histograms(lengths, 'Segment length $(\mu m)$', 'Frequency', 'Branch Length Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
            network_histograms(surfaces, 'Segment surface area $(\mu m)^2$', 'Frequency',
                               'Surface Area Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
            network_histograms(volumes, 'Segment volume $(\mu m)^3$', 'Frequency', 'Branch Volume Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
            network_histograms(radii, 'Segment radius $(\mu m)$', 'Frequency', 'Branch radii Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
            network_histograms(contractions, 'Segment contraction factor $\left(\\frac{displacement}{length}\\right)$',
                               'Frequency',
                               'Branch Contraction Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
            network_histograms(fractal_scores,
                               'Segment fractal dimension $\left(\\frac{ln(length)}{ln(displacement))}\\right)$',
                               'Frequency',
                               'Branch Fractal Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])

        if self.input_dic['SEGMENT_3D'] or self.input_dic['MESH_3D'] or self.input_dic['SKEL_3D']:
            img_enhanced, img_mask, img_skel, img_dist, img_original = \
                std_2d_segment(self.input_dic['TIF_FILE'],
                               scale_xy=(1, 1),
                               contrast_method=self.input_dic['CONTRAST_METHOD'],
                               thresh_method=self.input_dic['THRESH_METHOD'],
                               rwalk_thresh=(self.input_dic['RWALK_THRESH_LOW'],
                                             self.input_dic['RWALK_THRESH_HIGH']),
                               bth_k=self.input_dic['BTH_K_2D'],
                               wth_k=self.input_dic['WTH_K_2D'],
                               dila_gauss=self.input_dic['DILA_GAUSS'],
                               open_first=self.input_dic['OPEN_FIRST'],
                               open_k=self.input_dic['OPEN_K_2D'],
                               close_k=self.input_dic['CLOSE_K_2D'],
                               connected2D=self.input_dic['CONNECTED_3D_MASK'],
                               smooth=self.input_dic['SMOOTH'],
                               all_plots=False,
                               review_plot=False,
                               output_dir=self.input_dic['OUTPUT_DIR'],
                               name=self.name,
                               save_mask=False,
                               save_skel=False,
                               save_dist=False,
                               save_disp=False,
                               save_review=False,
                               generate_mesh_25=False,
                               connected_mesh=self.input_dic['CONNECTED_25D_MESH'],
                               connected_vol=self.input_dic['CONNECTED_25D_VOLUME'],
                               plot_25d=False,
                               save_volume_mask=False,
                               save_surface_meshes=False,
                               generate_volume_meshes=False)

            img_2d_stack = img_original.get_pages()
            scale_factor = img_original.downsample_factor

            img_dim = img_original.flattened.shape
            x_micron = scale_factor * self.input_dic['SCALE_X'] * img_dim[0]
            y_micron = scale_factor * self.input_dic['SCALE_Y'] * img_dim[1]
            units = np.ceil(max(x_micron, y_micron) / 750)
            img_original.set_units(units)

            print('\n\nScale factor: %f' % scale_factor)
            print('Units: %d um per pixel.' % units)

            std_3d_segment(img_2d_stack, img_mask,
                           scale=(self.input_dic['SCALE_X'] * scale_factor,
                                  self.input_dic['SCALE_Y'] * scale_factor,
                                  self.input_dic['SCALE_Z'] * 1),
                           units=units,
                           slice_contrast=self.input_dic['SLICE_CONTRAST'],
                           pre_thresh=self.input_dic['PRE_THRESH'],
                           maxr=self.input_dic['MAXR'],
                           minr=self.input_dic['MINR'],
                           h_pct_r=self.input_dic['H_PCT_R'],
                           thresh_pct=self.input_dic['THRESH_PCT'],
                           ellipsoid_method=self.input_dic['ELLIPSOID_METHOD'],
                           thresh_num_oct=self.input_dic['THRESH_NUM_OCT'],
                           thresh_num_oct_op=self.input_dic['THRESH_NUM_OCT_OP'],
                           max_iters=self.input_dic['MAX_ITERS'],
                           n_theta=self.input_dic['N_THETA'],
                           n_phi=self.input_dic['N_PHI'],
                           ray_trace_mode=self.input_dic['RAY_TRACE_MODE'],
                           theta=self.input_dic['THETA'],
                           max_escape=self.input_dic['MAX_ESCAPE'],
                           path_l=self.input_dic['PATH_L'],
                           bth_k=self.input_dic['BTH_K_3D'],
                           wth_k=self.input_dic['WTH_K_3D'],
                           window_size=(
                               self.input_dic['WINDOW_SIZE_Z'], self.input_dic['WINDOW_SIZE_X'],
                               self.input_dic['WINDOW_SIZE_Y']),
                           enforce_circular=self.input_dic['ENFORCE_ELLIPSOID_LUMEN'],
                           h_pct_ellip=self.input_dic['H_PCT_ELLIPSOID'],
                           fill_lumen_meshing=self.input_dic['FILL_LUMEN_MESHING'],
                           max_meshing_iters=self.input_dic['FILL_LUMEN_MESHING_MAX_ITS'],
                           squeeze_skel_blobs=self.input_dic['SQUEEZE_SKEL_BLOBS'] and self.input_dic['SKEL_3D'],
                           remove_skel_surf=self.input_dic['REMOVE_SKEL_SURF'] and self.input_dic['SKEL_3D'],
                           surf_tol=self.input_dic['SKEL_SURF_TOL'],
                           close_skels=self.input_dic['SKEL_CLOSING'] and self.input_dic['SKEL_3D'],
                           plot_slices=self.input_dic['PLOT_3D_THRESH_SLICES'] and self.input_dic['SEGMENT_3D'],
                           connected_3D=self.input_dic['CONNECTED_3D_MASK'],
                           plot_lumen_fill=self.input_dic['PLOT_LUMEN_FILL'] and self.input_dic['SEGMENT_3D'],
                           plot_3d=self.input_dic['PLOT_3D_MESHES'] and self.input_dic['MESH_3D'],
                           output_dir=self.input_dic['OUTPUT_DIR'],
                           name=self.name,
                           save_3d_masks=(self.input_dic['SAVE_3D_MASK'] or self.input_dic['VOLUME_ANALYSIS']) and
                                         self.input_dic['SEGMENT_3D'],
                           generate_meshes=self.input_dic['MESH_3D'],
                           generate_skels=self.input_dic['SKEL_3D'],
                           plot_skels=self.input_dic['PLOT_3D_SKELS'] and self.input_dic['SKEL_3D'],
                           save_skel=self.input_dic['SAVE_3D_SKEL'] and self.input_dic['SKEL_3D'],
                           save_surface_3D=self.input_dic['SAVE_3D_MESH'] and self.input_dic['MESH_3D'],
                           generate_volume_3D=self.input_dic['GENERATE_3D_VOLUME'] and self.input_dic['MESH_3D'],
                           save_surface_round=self.input_dic['SAVE_3D_MESH_ROUND'] and self.input_dic['MESH_3D'],
                           generate_volume_round=self.input_dic['GENERATE_3D_ROUND_VOLUME'] and self.input_dic[
                               'MESH_3D'])
