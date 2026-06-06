# 3D Reconstruction from Images — Photogrammetry

This project implements a Python pipeline for reconstructing 3D point clouds from multiple images using photogrammetry.

The project focuses mainly on **relative orientation**, where the camera poses are estimated from image correspondences. An **absolute orientation** pipeline is also included to validate the geometric reconstruction process using known calibration points.

---

## 1. Relative Orientation

Relative orientation reconstructs a 3D scene without using a known 3D reference frame. Instead, it estimates the relative positions of the cameras from matching points detected across several images.

The pipeline is:

1. Detect feature points in the images.
2. Match points between image pairs.
3. Filter incorrect matches using RANSAC.
4. Estimate the relative camera poses.
5. Triangulate 3D points.
6. Export and visualize the reconstructed point cloud.

In this project, SIFT was used for feature detection and matching, followed by geometric filtering and triangulation.

### Results

The relative-orientation pipeline was tested on several real scenes.

| Scene | Images | Reconstructed cameras | 3D points |
|---|---:|---:|---:|
| Pencil case | 18 | 18/18 | 13,182 |
| Pyramid | 23 | 23/23 | 19,388 |
| Stairs | 39 | 39/39 | 106,634 |
| Topographic map | 32 | 32/32 | 391,093 |

The pyramid reconstruction was evaluated after scale alignment. The average relative error was about **2.47%**, corresponding to an average absolute error of about **1.5 mm**.

### Example reconstructions

#### Pyramid

![Pyramid reconstruction](assets/relative_orientation/pyramid_result.png)

#### Stairs

![Stairs reconstruction](assets/relative_orientation/stairs_result.png)

#### Topographic map

![Topographic map reconstruction](assets/relative_orientation/topographic_map_result.png)

#### Error evaluation on the pyramid

![Pyramid error](assets/relative_orientation/pyramid_error.png)

---

## 2. Absolute Orientation

Absolute orientation was used as a validation step. In this setup, some 3D calibration points are known in advance, which makes it possible to estimate the camera projection matrices directly.

The pipeline is:

1. Select known 3D calibration points.
2. Click their corresponding 2D positions in each image.
3. Estimate the camera projection matrices using DLT.
4. Triangulate matched points in 3D.
5. Compare reconstructed control points with known measurements.

This part helped verify that the geometric core of the project — projection, calibration and triangulation — was working correctly.

LoFTR was also tested to improve point matching on difficult images, especially when SIFT produced unstable correspondences.

### Cube reconstruction

![Cube reconstruction](assets/absolute_orientation/cube_reconstruction.png)

### Altitude error on the cube

Four control points were selected on the top face of the cube. The altitude of the reconstructed points was compared with the theoretical height of the cube.

The maximum altitude error was about **1.03 mm**.

![Cube error heatmap](assets/absolute_orientation/cube_error_heatmap.png)

---

## Technologies

Python, OpenCV, NumPy, SciPy, Matplotlib, SIFT, RANSAC, LoFTR, CloudCompare

---

## Project context

Academic project at IMT Atlantique on 3D reconstruction by photogrammetry.


