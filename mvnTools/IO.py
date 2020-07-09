import ntpath
import os
import pickle

from mvnTools.Network2d import Network2d
from mvnTools.Network2dTools import network_histograms
from mvnTools.Pipelines import std_2d_segment, network_2d_analysis


class IO:
    name = None
    input_dic = {
        # Input image and scale values (um / pixel)
        'TIF_FILE': None,
        'SCALE_X': None,
        'SCALE_Y': None,

        # Piplelines to Run
        'SEGMENT_2D': False,
        'NETWORK_2D_GEN': False,
        'NETWORK_2D_COMPARE': False,

        # Save Parameters
        'OUTPUT_DIR': None,

        'SAVE_2D_MASK': True,
        'SAVE_2D_SKEL': True,
        'SAVE_2D_DIST': True,
        'SAVE_2D_DISPLAY': True,
        'SAVE_2D_REVIEW': True,

        'SAVE_2D_NETWORK': True,

        # Display Parameters
        'PLOT_ALL_2D': False,
        'REVIEW_PLOT_2D': False,

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

        # Piplelines to Run
        'SEGMENT_2D=': bool,
        'MESH_25D=': bool,
        'NETWORK_2D_GEN=': bool,
        'NETWORK_2D_COMPARE=': bool,

        # Save Parameters
        'OUTPUT_DIR=': str,

        'SAVE_2D_MASK=': bool,
        'SAVE_2D_SKEL=': bool,
        'SAVE_2D_DIST=': bool,
        'SAVE_2D_DISPLAY=': bool,
        'SAVE_2D_REVIEW=': bool,

        'SAVE_2D_NETWORK=': bool,

        # Display Parameters
        'PLOT_ALL_2D=': bool,
        'REVIEW_PLOT_2D=': bool,

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
            self.input_dic['PLOT_25D_MESH'] = False
            self.input_dic['PLOT_NETWORK_GEN'] = False
            self.input_dic['PLOT_NETWORK_DATA'] = False

        if all_plots_mode:
            self.input_dic['PLOT_ALL_2D'] = True
            self.input_dic['PLOT_25D_MESH'] = True
            self.input_dic['REVIEW_PLOT_2D'] = True
            self.input_dic['PLOT_NETWORK_GEN'] = True
            self.input_dic['PLOT_NETWORK_DATA'] = True

        if echo_inputs:
            for ins in self.input_dic:
                print(ins + '=' + str(self.input_dic[ins]))

        if self.input_dic['SEGMENT_2D'] or self.input_dic['MESH_25D']:
            std_2d_segment(self.input_dic['TIF_FILE'],
                           scale_xy=(self.input_dic['SCALE_X'], self.input_dic['SCALE_Y']),
                           scale_z=self.input_dic['SCALE_Z'],
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
                           save_review=self.input_dic['SAVE_2D_REVIEW'] and self.input_dic['SEGMENT_2D']
                           )

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
                               save_review=False
                               )

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
            save_path = ''
            if self.input_dic['SAVE_2D_NETWORK']:
                if not os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks/'):
                    os.mkdir(self.input_dic['OUTPUT_DIR'] + 'networks/')
                if not os.path.isdir(self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/'):
                    os.mkdir(self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/')
                save_path = self.input_dic['OUTPUT_DIR'] + 'networks/' + self.name + '/'

                save_obj = save_path + 'network_object.gpickle'
                with open(save_obj, 'wb') as network:
                    pickle.dump(network_object, network)

            network_2d_analysis(network_object, output_dir=save_path, save_outs=self.input_dic['SAVE_2D_NETWORK'],
                                plot_outs=self.input_dic['PLOT_NETWORK_DATA'], name=self.name)

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

            lengths, volumes, surfaces, radii, contractions, fractal_scores, connect = [], [], [], [], [], [], []
            print('Comparing %d networks' % len(graph_objs))
            for g in graph_objs:
                lengths.append(g.lengths)
                volumes.append(g.volumes)
                surfaces.append(g.surfaces)
                radii.append(g.radii)
                contractions.append(g.contractions)
                fractal_scores.append(g.fractal_scores)
                connect.append(g.connectivity)

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
            network_histograms(radii, 'Segment radius $(\mu m)$', 'Frequency', 'Branch Radii Distribution',
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

            network_histograms(connect,
                               'Connectivity',
                               'Frequency',
                               'Node Connectivity Distribution',
                               names, save=True, ouput_dir=self.input_dic['OUTPUT_DIR'] + 'networks-compare/',
                               show=self.input_dic['PLOT_NETWORK_DATA'])
