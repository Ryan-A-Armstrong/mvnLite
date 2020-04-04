# mvn-analysis
mvn-analysis provides image processing and analysis pipelines for .tif stack images of 3-dimensional microvascular networks (MVNs). 

Specifically, images of MVNs produced in the [Polacheck Lab](https://www.polachecklaboratory.com/ "Polacheck Group Page") are segmented in 2 and 3 dimensions to create water-tight triangle surface meshes and fully connected tetrahedral lumen meshes. These meshes can then be used as structural inputs for fluid dynamics models developed by the [Griffith Lab](http://griffith.web.unc.edu/ "Griffith Group Page") and [Dr. Simone Rossi](https://github.com/rossisimone/beatit "Rossi Github Repository") to estimate fluid forces within the MVN. Further, vascularly relevant statistics are applied to the 2D/3D segmentations and the segmentations are converted to NetworkX graphs with nodes at branches and endpoints and edges weighted by vessel length and width.

## Dependancies
TODO

## Running
```
python mvn.py -h
usage: mvn.py [-h] [-f INPUT_FILE] [-d INPUT_DIR] [-s SILENT] [-e ECHO_INPUTS]

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input-file INPUT_FILE
                        string path to input file
  -d INPUT_DIR, --input-dir INPUT_DIR
                        string path to input directory (batch runs all .txt files)
  -s SILENT, --silent SILENT
                        1 to suppress all prints (useful in batch mode)
  -e ECHO_INPUTS, --echo-inputs ECHO_INPUTS
                        1 to print the inputs being used
```
## Input Template .txt
```
# MVN Image Analysis Input File Template

===============
Required Fields
===============

# Input image and scale values (um / pixel)

TIF_FILE=
SCALE_X=
SCALE_Y=
SCALE_Z=

#### Piplelines to Run ####
SEGMENT_2D=0
MESH_25D=0
SEGMENT_3D=0
MESH_3D=0
NETWORK_2D=0
NETOWRK_3D=0

#### Save Parameters #####

OUTPUT_DIR=

SAVE_2D_MASK=1
SAVE_2D_SKEL=1
SAVE_2D_DIST=1
SAVE_2D_DISPLAY=1
SAVE_2D_REVIEW=1

SAVE_25D_MESH=1
GENERATE_25D_VOLUME=0

SAVE_3D_MASK=1
SAVE_3D_SKEL=1

SAVE_3D_MESH=1
GENERATE_3D_VOLUME=0
SAVE_3D_MESH_ROUND=1
GENERATE_3D_ROUND_VOLUME=0

SAVE_2D_NETWORK=1
SAVE_3D_NETWORK=1


#### Display Parameters #####

PLOTS_2D=0
REVIEW_PLOT_2D=0
PLOT_SLICES=0
PLOT_LUMEN_FILL=0
PLOT_3D=0


=====================
Adjustable Parameters
=====================

#### 2D Analysis #####

# original, rescale, equalize, adaptive
CONTRAST_METHOD=rescale

# entropy, random-walk
THRESH_METHOD=entropy

RWALK_THRESH_LOW=0.75
RWALK_THRESH_HIGH=0.8
BTH_K_2D=3
WTH_K_2D=3
DILA_GAUSS=0.333
OPEN_FIRST=0
OPEN_K_2D=0
CLOSE_K_2D=5
CONNECTED_2D=1
SMOOTH=1

CONNECTED_25D=1


#### GRAPH Analysis #####

NEAR_NODE_TOL=5
LENGTH_TOL=1

#### 3D Analysis #####
# original, rescale, equalize, adaptive
SLICE_CONTRAST=original

# sauvola, none
PRE_THRESH=sauvola

BTH_K_3D=3
WTH_K_3D=3
OPEN_K_3D=0
CLOSE_K_3D=5

# octants, pairs, ball
ELLIPSOID_METHOD=octants

MAXR=15
MINR=1
H_PCT_R=0.5
THRESH_PCT=0.15
THRESH_NUM_OCT=5
THRESH_NUM_OCT_OP=3
MAX_ITERS=1

# uniform, random
RAY_TRACE_MODE=uniform

# sweep, xy, exclude_z_pole
THETA=exclude_z_pole

N_THETA=6
N_PHI=6
MAX_ESCAPE=2
PATH_L=1

FILL_LUMEN_MESHING=1
FILL_LUMEN_MESHING_MAX_ITS=3

ENFORCE_ELLIPSOID_LUMEN=1
H_PCT_ELLIPSOID=1.00

WINDOW_SIZE_X=15
WINDOW_SIZE_Y=15
WINDOW_SIZE_Z=7

```

## Examples and explaination of input file
TODO
