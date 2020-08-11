# mvnLite
mvnLite provides image processing and analysis pipelines for .tif stack images of microvascular networks (MVNs). It is a more stable subset of mvn-analysis (a 2.5D and 3D extension), however neither are robustly validated to be used for research publications. 

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


