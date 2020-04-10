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

![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/09092019_gel_region_10x-2um-pix.png)
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/original_sample-1um-pix.png)
### 2.5D Surface and Volume Meshes (3D Geometries from 2D Data)
Lumens are forced to ellipsoidal cross-sections. The ellipsoid axes ratio can either be specified or can be automatically determined to fit the calculated z-depth of the image data.

Volumes are guaranteed to have no internal holes. Surfaces are guaranteed to be watertight.

#### Surface mesh
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/25d-10x.png) 
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/25d-original.png)

#### Volume mesh
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/original-volume.png)

### 3D Surface Meshes,Volume Meshes, and Skeletonizations
Lumens are segmented at each image plane and interpolated to fill the z-step of the confocal imaging. A combination of iterative local thresholding, ray tracing, and meshing is used to fill the lumen segmentation (the original images are only of the surface of the vessels).

Volumes are guaranteed to have no internal holes. Surfaces are most likely watertight, but would need an eternal “shrink wrapping” to guarantee.

![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/3d-10x.png)

```
original_sample volume analysis sample
==============================================================================
Binary image generated from 25d.
Connected is True

(43 um, 635 um, 446 um)	Bounding box dimensions (z, x, y)
2003360 um^3	Total volume
0.164506	Vascular density (volume vessel/volume space)
26.717513 um	Average distance from vessel wall
132.698907 um	Maximum distance from vessel wall
131.654768 um	Average maximum distance from vessel wall per z-plane

Average distance to vessel wall per z-plane (rounded to um):
[27 27 26 26 26 25 25 25 25 25 25 25 25 26 26 27 27 28 28 29 29 29 29 29
 29 28 28 27 27 26 26 25 25 25 25 25 25 25 25 26 26 26 27]


Maximum distance to vessel wall per z-plane (rounded to um):
[132 132 132 132 132 132 131 131 131 131 131 131 131 131 131 131 131 131
 131 131 131 131 131 131 131 131 131 131 131 131 131 131 131 131 131 131
 131 131 132 132 132 132 132]

==============================================================================

==============================================================================
Binary image generated from 3D.
Connected is True

(40 um, 635 um, 469 um)	Bounding box dimensions (z, x, y)
2662724 um^3	Total volume
0.223522	Vascular density (volume vessel/volume space)
27.234602 um	Average distance from vessel wall
143.209637 um	Maximum distance from vessel wall
140.802669 um	Average maximum distance from vessel wall per z-plane

Average distance to vessel wall per z-plane (rounded to um):
[26 26 25 25 25 26 26 26 27 27 27 27 27 27 28 28 28 28 28 28 29 29 29 29
 29 29 29 29 29 29 28 28 27 26 25 24 24 24 24 24]


Maximum distance to vessel wall per z-plane (rounded to um):
[139 139 139 139 139 139 139 139 139 139 139 139 139 139 139 140 140 140
 140 140 140 140 140 140 140 140 141 141 141 141 141 141 142 142 142 142
 142 142 143 143]

==============================================================================

==============================================================================
Binary image generated from enforce-ellip-0.5000.
Connected is True

(40 um, 635 um, 469 um)	Bounding box dimensions (z, x, y)
2662724 um^3	Total volume
0.223522	Vascular density (volume vessel/volume space)
27.234602 um	Average distance from vessel wall
143.209637 um	Maximum distance from vessel wall
140.802669 um	Average maximum distance from vessel wall per z-plane

Average distance to vessel wall per z-plane (rounded to um):
[26 26 25 25 25 26 26 26 27 27 27 27 27 27 28 28 28 28 28 28 29 29 29 29
 29 29 29 29 29 29 28 28 27 26 25 24 24 24 24 24]


Maximum distance to vessel wall per z-plane (rounded to um):
[139 139 139 139 139 139 139 139 139 139 139 139 139 139 139 140 140 140
 140 140 140 140 140 140 140 140 141 141 141 141 141 141 142 142 142 142
 142 142 143 143]

==============================================================================
```

### 2D Weighted Networks
NetworkX graph representation of the 2D segmentation.

![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/network-10x.png)
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/network-original.png)


```
09092019_gel_region_10x Output Example
98	Number of branch points
31	Number of end points
9256.2550	Total length um
1086720.9267	Total surface area um^2 (assumes circular vessels)
14924270.7128	Total volume um^3 (assumes circular vessels)
56.0985	Average branch length
90450.1255	Average branch volume
0.9118	Average contraction factor

1.6588	Average node connectivity
```

```
original_sample directionality summary

2D Directionality Summary. Weighted is False.
  Percent Vertical:	 0.3188
  Percent Horizontal:	 0.2514
  Percent L. Diagonal:	 0.2340
  Percent R. Diagonal:	 0.1958

  Z-Score of max:	 1.5452

2D Directionality Summary. Weighted is True.
  Percent Vertical:	 0.3334
  Percent Horizontal:	 0.2421
  Percent L. Diagonal:	 0.2052
  Percent R. Diagonal:	 0.2192

  Z-Score of max:	 1.6708
```

#### Compare Histograms
![alt text](https://github.com/Ryan-A-Armstrong/mvn-analysis/blob/master/example_images/hist-summary.png)

### 3D Weighted Networks
In progress
