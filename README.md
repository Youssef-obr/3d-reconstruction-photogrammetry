# 3D Reconstruction from Images — Photogrammetry

This project implements a Python pipeline for reconstructing 3D point clouds from multiple images using photogrammetry.

The main focus is **relative orientation**, where camera poses are estimated from image correspondences, without requiring a known 3D reference frame. An **absolute orientation** part is also included to validate the geometric reconstruction pipeline using known calibration points.

---

## 1. Relative Orientation

Relative orientation reconstructs a 3D scene by estimating the relative positions of the cameras from matching points detected across multiple images.

The pipeline follows these steps:

* Feature detection and matching between images
* Geometric filtering with RANSAC
* Fundamental and essential matrix estimation
* Relative camera pose recovery
* 3D point triangulation
* Point cloud export and visualization

### Geometric principle

For two matching image points `x1` and `x2`, the epipolar constraint is:

```text
x2ᵀ F x1 = 0
```

where `F` is the fundamental matrix.

In practice, some matches are wrong. RANSAC is used to estimate `F` robustly by keeping only the correspondences that are geometrically consistent.

RANSAC solves the following robust selection problem:

```text
F* = argmax_F #{ i : d(x2_i, F x1_i) < τ }
```

where `d` is the epipolar error and `τ` is an inlier threshold.

A common error used for this is the Sampson distance:

```text
d(x2, F x1) =
(x2ᵀ F x1)²
/
((F x1)_1² + (F x1)_2² + (Fᵀ x2)_1² + (Fᵀ x2)_2²)
```

After estimating `F`, the essential matrix is computed using the camera intrinsic matrix `K`:

```text
E = Kᵀ F K
```

The relative rotation and translation between cameras are then recovered from `E`, and 3D points are reconstructed by triangulation.

## Results

The method was tested on several real scenes, including a pyramid, stairs and a topographic map.

| Scene           | Images | Reconstructed cameras | 3D points |
| --------------- | -----: | --------------------: | --------: |
| Pyramid         |     23 |                 23/23 |    19,388 |
| Stairs          |     39 |                 39/39 |   106,634 |
| Topographic map |     32 |                 32/32 |   391,093 |

The pyramid reconstruction was evaluated after scale alignment. The average relative error was about **2.47%**, corresponding to an average absolute error of about **1.5 mm**.

## Example reconstructions

### Pyramid

<p align="center">
  <img src="assets/relative_orientation/pyramid_3D.png" width="480">
</p>

### Stairs

<p align="center">
  <img src="assets/relative_orientation/stairs_3D.png" width="480">
</p>

### Topographic map — top view

<p align="center">
  <img src="assets/relative_orientation/map_3D.png" width="480">
</p>

### Topographic map — side view

<p align="center">
  <img src="assets/relative_orientation/side_map_3D.png" width="480">
</p>

---

## 2. Absolute Orientation

Absolute orientation was used to validate the geometric part of the reconstruction pipeline. In this case, known 3D calibration points are used to estimate camera projection matrices.

The camera projection model is:

```text
x ~ P X
```

where `X` is a 3D point, `x` is its 2D image projection, and `P` is the camera projection matrix.

The projection matrix `P` is estimated using DLT from known 3D calibration points and their corresponding 2D image positions. This allows the reconstruction to be expressed directly in a known metric frame.

This part helped verify the main geometric steps:

* Projection
* Camera calibration
* Triangulation
* Reconstruction in a known metric frame

LoFTR was also tested to improve point matching on more difficult images, especially when classical local matching was unstable.

For the cube experiment, four control points were selected on the top face. Their reconstructed altitude was compared with the theoretical height of the cube:

```text
e_z = z_rec - z_true
```

The maximum altitude error was about **1.03 mm**.

---

## Technologies

Python, OpenCV, NumPy, SciPy, Matplotlib, SIFT, RANSAC, LoFTR, CloudCompare

---

## Project context

Academic project at IMT Atlantique on 3D reconstruction by photogrammetry.
