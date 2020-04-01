from mvnTools.Pipelines import std_2d_segment, std_3d_segment, generate_2d_network, segment_2d_to_meshes


class IO:
    input_dic = {'TIF_FILE': None,
                 'SCALE_X': None,
                 'SCALE_Y': None,
                 'SCALE_Z': None,
                 'OUTPUT_DIR': None,
                 'FILES_2D': True,
                 'MESH_25D': True,
                 'MESH_3D': True,
                 'SKEL_3D': True,
                 'NETWORK': True,
                 'NETWORK_ANALYSIS': True,
                 'PLOTS_2D': True,
                 'REVIEW_PLOT_2D': True,
                 'PLOT_SLICES': False,
                 'PLOT_LUMEN_FILL': True,
                 'PLOT_3D': True,
                 'CONTRAST_METHOD': 'rescale',
                 'THRESH_METHOD': 'entropy',
                 'RWALK_THRESH_LOW': 0,
                 'RWALK_THRESH_HIGH': 0,
                 'BTH_K_2D': 3,
                 'WTH_K_2D': 3,
                 'DILA_GAUSS': 0.333,
                 'OPEN_FIRST': 0,
                 'OPEN_K_2D': 0,
                 'CLOSE_K_2D': 5,
                 'CONNECTED_2D': 0,
                 'SMOOTH': 1,
                 'NEAR_NODE_TOL': 5,
                 'LENGTH_TOL': 1,
                 'BTH_K_3D': 3,
                 'WTH_K_3D': 3,
                 'OPEN_K_3D': 0,
                 'CLOSE_K_3D': 5,
                 'ELLIPSOID_METHOD': 'octants',
                 'MAXR': 15,
                 'MINR': 1,
                 'H_PCT_R': 0.75,
                 'THRESH_PCT': 0.15,
                 'THRESH_NUM_OCT': 5,
                 'THRESH_NUM_OCT_OP': 3,
                 'MAX_ITERS': 1,
                 'RAY_TRACE_MODE': 'uniform',
                 'THETA': 'sweep',
                 'N_THETA': 6,
                 'N_PHI': 6,
                 'MAX_ESCAPE': 1,
                 'PATH_L': 1,
                 'WINDOW_SIZE_X': 15,
                 'WINDOW_SIZE_Y': 15,
                 'WINDOW_SIZE_Z': 7}

    input_dic_setter = {'TIF_FILE=': str,
                        'SCALE_X=': float,
                        'SCALE_Y=': float,
                        'SCALE_Z=': float,
                        'OUTPUT_DIR=': str,
                        'FILES_2D=': bool,
                        'MESH_25D=': bool,
                        'MESH_3D=': bool,
                        'SKEL_3D=': bool,
                        'NETWORK=': bool,
                        'NETWORK_ANALYSIS=': bool,
                        'PLOTS_2D=': bool,
                        'REVIEW_PLOT_2D=': bool,
                        'PLOT_SLICES=': bool,
                        'PLOT_LUMEN_FILL=': bool,
                        'PLOT_3D=': bool,
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
                        'NEAR_NODE_TOL=': int,
                        'LENGTH_TOL=': int,
                        'BTH_K_3D=': int,
                        'WTH_K_3D=': int,
                        'OPEN_K_3D=': int,
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
                        'WINDOW_SIZE_X=': int,
                        'WINDOW_SIZE_Y=': int,
                        'WINDOW_SIZE_Z=': int}

    def __init__(self, input_file, echo_inputs=False):
        with open(input_file, 'r') as f:
            inputs = f.read()

            for param in self.input_dic_setter:
                index_param = inputs.index(param)
                index_end_line = inputs.index('\n', index_param)
                val = inputs[index_param + len(param):index_end_line]

                if self.input_dic_setter[param] is str:
                    inputs[index_param + len(param):index_end_line]

                elif self.input_dic_setter[param] is int:
                    val = int(val)

                elif self.input_dic_setter[param] is float:
                    val = float(val)

                elif self.input_dic_setter[param] is bool:
                    val = bool(int(val))

                self.input_dic[param[:-1]] = val

        f.close()
        if echo_inputs:
            for ins in self.input_dic:
                print(ins + '='  + str(self.input_dic[ins]))

        if self.input_dic['FILES_2D']:
            img_enhanced, img_mask, img_skel, img_dist, img_original = \
                std_2d_segment(self.input_dic['TIF_FILE'],
                               (self.input_dic['SCALE_X'],
                                self.input_dic['SCALE_Y']),
                               contrast_method=self.input_dic[
                                   'CONTRAST_METHOD'],
                               thresh_method=self.input_dic[
                                   'THRESH_METHOD'],
                               rwalk_thresh=(self.input_dic['RWALK_THRESH_LOW'],
                                             self.input_dic['RWALK_THRESH_HIGH']),
                               bth_k=self.input_dic['BTH_K_2D'],
                               wth_k=self.input_dic['WTH_K_2D'],
                               dila_gauss=self.input_dic[
                                   'DILA_GAUSS'],
                               open_first=self.input_dic[
                                   'OPEN_FIRST'],
                               open_k=self.input_dic[
                                   'OPEN_K_2D'],
                               close_k=self.input_dic[
                                   'CLOSE_K_2D'],
                               connected=self.input_dic[
                                   'CONNECTED_2D'],
                               smooth=self.input_dic['SMOOTH'],
                               all_plots=self.input_dic[
                                   'PLOTS_2D'],
                               review_plot=self.input_dic[
                                   'REVIEW_PLOT_2D'])

        if self.input_dic['NETWORK']:
            img_enhanced, img_mask, img_skel, img_dist, img_original = \
                std_2d_segment(self.input_dic['TIF_FILE'],
                               (self.input_dic['SCALE_X'],
                                self.input_dic['SCALE_Y']),
                               contrast_method=self.input_dic[
                                   'CONTRAST_METHOD'],
                               thresh_method=self.input_dic[
                                   'THRESH_METHOD'],
                               rwalk_thresh=(self.input_dic[
                                                 'RWALK_THRESH_LOW'],
                                             self.input_dic[
                                                 'RWALK_THRESH_HIGH']),
                               bth_k=self.input_dic['BTH_K_2D'],
                               wth_k=self.input_dic['WTH_K_2D'],
                               dila_gauss=self.input_dic[
                                   'DILA_GAUSS'],
                               open_first=self.input_dic[
                                   'OPEN_FIRST'],
                               open_k=self.input_dic[
                                   'OPEN_K_2D'],
                               close_k=self.input_dic[
                                   'CLOSE_K_2D'],
                               connected=self.input_dic[
                                   'CONNECTED_2D'],
                               smooth=self.input_dic['SMOOTH'],
                               all_plots=self.input_dic[
                                   'PLOTS_2D'],
                               review_plot=self.input_dic[
                                   'REVIEW_PLOT_2D'])

            G = generate_2d_network(img_skel, img_dist,
                                    near_node_tol=self.input_dic['NEAR_NODE_TOL'],
                                    length_tol=self.input_dic['LENGTH_TOL'],
                                    img_enhanced=img_enhanced,
                                    plot=self.input_dic['PLOT_3D'])

        if self.input_dic['MESH_25D']:
            mesh = segment_2d_to_meshes(img_dist, img_skel,
                                                      all_plots=self.input_dic['PLOT_3D'])

        if self.input_dic['MESH_3D']:
            img_enhanced, img_mask, img_skel, img_dist, img_original = \
                std_2d_segment(self.input_dic['TIF_FILE'],
                               (1, 1),
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
                               connected=True,
                               smooth=self.input_dic['SMOOTH'],
                               all_plots=False,
                               review_plot=self.input_dic['REVIEW_PLOT_2D'])

            std_3d_segment(img_original.get_pages(), img_mask,
                           (self.input_dic['SCALE_X'], self.input_dic['SCALE_Y'], self.input_dic['SCALE_Z']),
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
                           window_size=(self.input_dic['WINDOW_SIZE_Z'], self.input_dic['WINDOW_SIZE_X'],
                                        self.input_dic['WINDOW_SIZE_Y']),
                           plot_slices=self.input_dic['PLOT_SLICES'],
                           plot_lumen_fill=self.input_dic['PLOT_LUMEN_FILL'],
                           plot_3d=self.input_dic['PLOT_3D'])
