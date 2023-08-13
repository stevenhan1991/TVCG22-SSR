# SSR-TVD: Spatial Super-Resolution for Time-Varying Data Analysis and Visualization
Pytorch implementation for SSR-TVD: Spatial Super-Resolution for Time-Varying Data Analysis and Visualization.

## Prerequisites
- Linux
- CUDA >= 10.0
- Skimage
- Python >= 3.7
- Numpy
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis. The low-resolution volumes are obtained by applying bicubic interpolation to high-resolution ones.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset 'Vortex'
```

- inference
```
python3 main.py --mode 'inf'
```

## Citation 
```
@article{Han-TVCG21,
	Author = {J. Han and C. Wang},
	Journal = {IEEE Transactions on Visualization and Computer Graphics},
	Number = {6},
	Pages = {2445-2456},
	Title = {{SSR-TVD}: Spatial Super-Resolution for Time-Varying Data Analysis and Visualization},
	Volume = {28},
	Year = {2022}}

```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CNS-1629914, DUE-1833129, and IIS-1955395.
