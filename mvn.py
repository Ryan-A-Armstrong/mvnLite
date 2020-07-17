import argparse
import glob
import os

from mvnTools.IO import IO

# Test commit

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input-file', type=str, required=False,
                    help='string path to input file')
parser.add_argument('-d', '--input-dir', type=str, required=False,
                    help='string path to input directory (batch runs all .txt files)')
parser.add_argument('-s', '--silent', type=int, required=False,
                    help='1 to suppress all plots (useful in batch mode)')
parser.add_argument('-p', '--plot-all', type=int, required=False,
                    help='1 to show all plots (useful to adjust parameters)')
parser.add_argument('-e', '--echo-inputs', type=int, required=False,
                    help='1 to print the inputs being used')

args = vars(parser.parse_args())
args['silent'] = bool(args['silent'])
args['plot_all'] = bool(args['plot_all'])
args['echo_inputs'] = bool(args['echo_inputs'])

if args['input_file'] is not None:
    IO(args['input_file'], echo_inputs=args['echo_inputs'], silent_mode=args['silent'], all_plots_mode=args['plot_all'])

if args['input_dir'] is not None:
    n_files = len(glob.glob(os.path.join(args['input_dir'], '*.txt')))
    c_file = 1
    for filename in glob.glob(os.path.join(args['input_dir'], '*.txt')):
        print('\n\nProcessing image %d of %d' % (c_file, n_files))
        IO(filename, echo_inputs=args['echo_inputs'], silent_mode=args['silent'], all_plots_mode=args['plot_all'])
        c_file += 1
