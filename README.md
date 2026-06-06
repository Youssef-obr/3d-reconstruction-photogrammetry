# 3D Reconstruction from Images — Photogrammetry

This project implements a Python pipeline for reconstructing 3D point clouds from multiple images using photogrammetry.

The main focus is **relative orientation**, where camera poses are estimated from image correspondences, without requiring a known 3D reference frame. An **absolute orientation** part is also included to validate the geometric reconstruction pipeline using known calibration points.

---

## 1. Relative Orientation

Relative orientation reconstructs a 3D scene by estimating the relative positions of the cameras from matching points detected across multiple images.

The pipeline follows these steps:

* Feature detection and matching between images
* Geometric filtering with RANSAC
* Camera pose estimation
* 3D point triangulation
* Point cloud export and visualization

The method was tested on several real scenes, including a pyramid, stairs and a topographic map.

## Results

| Scene           | Images | Reconstructed cameras | 3D points |
| --------------- | -----: | --------------------: | --------: |
| Pyramid         |     23 |                 23/23 |    19,388 |
| Stairs          |     39 |                 39/39 |   106,634 |
| Topographic map |     32 |                 32/32 |   391,093 |

The pyramid reconstruction was evaluated after scale alignment. The average relative error was about **2.47%**, corresponding to an average absolute error of about **1.5 mm**.

## Example reconstructions

### Pyramid

![Pyramid reconstruction](assets/relative_orientation/pyramid_3D.png)

### Stairs

![Stairs reconstruction](assets/relative_orientation/stairs_3D.png)

### Topographic map — top view

![Topographic map reconstruction](assets/relative_orientation/map_3D.png)

### Topographic map — side view

![Topographic map side view](assets/relative_orientation/side_map_3D.png)

---

## 2. Absolute Orientation

Absolute orientation was used to validate the geometric part of the reconstruction pipeline. In this case, known 3D calibration points are used to estimate camera projection matrices.

This part helped verify the main geometric steps:

* Projection
* Camera calibration
* Triangulation
* Reconstruction in a known metric frame

LoFTR was also tested to improve point matching on more difficult images, especially when classical local matching was unstable.

For the cube experiment, four control points were selected on the top face. Their reconstructed altitude was compared with the theoretical height of the cube, giving a maximum altitude error of about **1.03 mm**.

---

## Technologies

Python, OpenCV, NumPy, SciPy, Matplotlib, SIFT, RANSAC, LoFTR, CloudCompare

---

## Project context

Academic project at IMT Atlantique on 3D reconstruction by photogrammetry.
