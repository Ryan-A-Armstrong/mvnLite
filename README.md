# mvn-analysis
mvn-analysis provides image processing and analysis pipelines for .tif stack images of 3-dimensional microvascular networks (MVNs). 

Specifically, images of MVNs produced in the [Polacheck Lab](https://www.polachecklaboratory.com/ "Polacheck Group Page") are segmented in 2 and 3 dimensions to create water-tight triangle surface meshes and fully connected tetrahedral lumen meshes. These meshes can then be used as structural inputs for fluid dynamics models developed by the [Griffith Lab](http://griffith.web.unc.edu/ "Griffith Group Page") and [Dr. Simone Rossi](https://github.com/rossisimone/beatit "Rossi Github Repository") to estimate fluid forces within the MVN. Further, vascularly relevant statistics are applied to the 2D/3D segmentations and the segmentations are converted to NetworkX graphs with nodes at branches and endpoints and edges weighted by vessel length and width.

## Dependancies
```
$ pip install -r requirements.txt
```

## Running
```
python mvn.py -h
usage: mvn.py [-h] [-f INPUT_FILE] [-d INPUT_DIR] [-s SILENT] [-p PLOT_ALL]
              [-e ECHO_INPUTS]

optional arguments:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input-file INPUT_FILE
                        string path to input file
  -d INPUT_DIR, --input-dir INPUT_DIR
                        string path to input directory (batch runs all .txt files)
  -s SILENT, --silent SILENT
                        1 to suppress all plots (useful in batch mode)
  -p PLOT_ALL, --plot-all PLOT_ALL
                        1 to show all plots (useful to adjust parameters)
  -e ECHO_INPUTS, --echo-inputs ECHO_INPUTS
                        1 to print the inputs being used

```

## Examples of major outputs
Scale bars default to 100 microns.
### 2D Segmentations (Binary Masks, Distance Transforms, Skeletons)
Can retain either the entire image region or the largest connected region. 

### 2.5D Surface and Volume Meshes (3D Geometries from 2D Data)
Lumens are forced to ellipsoidal cross-sections. The ellipsoid axes ratio can either be specified or can be automatically determined to fit the calculated z-depth of the image data.

Volumes are guaranteed to have no internal holes. Surfaces are guaranteed to be watertight. 

### 3D Surface Meshes,Volume Meshes, and Skeletonizations
Lumens are segmented at each image plane and interpolated to fill the z-step of the confocal imaging. A combination of iterative local thresholding, ray tracing, and meshing is used to fill the lumen segmentation (the original images are only of the surface of the vessels).

Volumes are guaranteed to have no internal holes. Surfaces are most likely watertight, but would need an eternal “shrink wrapping” to guarantee.

### 2D Weighted Networks
NetworkX graph representation of the 2D segmentation.

### 3D Weighted Networks
In progress
