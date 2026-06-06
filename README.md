# 3D Reconstruction from Images — Photogrammetry

This project implements a Python pipeline for reconstructing 3D point clouds from multiple images using photogrammetry.

The project is mainly focused on **relative orientation**, where the camera poses are estimated from correspondences between images, without using a known 3D reference frame. A second part based on **absolute orientation** is also included to validate the geometric reconstruction pipeline using known calibration points.

---

## 1. Relative Orientation

Relative orientation aims to reconstruct a 3D scene from several images by estimating the relative positions and orientations of the cameras.

Unlike absolute orientation, the 3D reference frame of the object is not known in advance. The reconstruction is obtained only from correspondences between images.

The pipeline follows these steps:

* Detect and match feature points between images
* Remove inconsistent matches using RANSAC
* Estimate the fundamental matrix
* Compute the essential matrix using camera intrinsics
* Recover the relative camera poses
* Triangulate 3D points
* Export and visualize the reconstructed point cloud

---

## Geometric Principle

For two images, a 3D point is projected into two corresponding image points. In homogeneous coordinates, these image points are written as:

$$
\mathbf{x}_1 =
\begin{pmatrix}
u_1 \
v_1 \
1
\end{pmatrix},
\qquad
\mathbf{x}_2 =
\begin{pmatrix}
u_2 \
v_2 \
1
\end{pmatrix}.
$$

The two corresponding points must satisfy the epipolar constraint:

$$
\mathbf{x}_2^{\top}\mathbf{F}\mathbf{x}_1 = 0.
$$

Here, $\mathbf{F}$ is the fundamental matrix. It describes the projective geometric relation between two camera views.

In practice, some feature matches are wrong. RANSAC is used to robustly estimate $\mathbf{F}$ while rejecting outliers. It repeatedly samples small sets of matches, estimates a candidate fundamental matrix, and keeps the model that explains the largest number of consistent correspondences.

A simple way to write this is:

$$
\mathbf{F}^{\star}
==================

\arg\max_{\mathbf{F}}
N_{\mathrm{inliers}}(\mathbf{F}).
$$

where:

* $\mathbf{F}^{\star}$ is the selected fundamental matrix;
* $N_{\mathrm{inliers}}(\mathbf{F})$ is the number of matches that satisfy the epipolar constraint;
* matches with a geometric error below a threshold $\tau$ are kept as inliers;
* the other matches are rejected as outliers.

Once the fundamental matrix is estimated, the essential matrix is computed using the intrinsic calibration matrix $\mathbf{K}$:

$$
\mathbf{E} = \mathbf{K}^{\top}\mathbf{F}\mathbf{K}.
$$

The relative rotation and translation between the two cameras are then recovered from $\mathbf{E}$. Finally, the 3D points are reconstructed by triangulation.

---

## Relative Orientation Results

The relative-orientation pipeline was tested on several real scenes.

| Scene           | Images | Reconstructed cameras | 3D points |
| --------------- | -----: | --------------------: | --------: |
| Pyramid         |     23 |                 23/23 |    19,388 |
| Stairs          |     39 |                 39/39 |   106,634 |
| Topographic map |     32 |                 32/32 |   391,093 |

The pyramid reconstruction was evaluated after scale alignment. The average relative error was about **2.47%**, corresponding to an average absolute error of about **1.5 mm**.

---

## Example Reconstructions

### Pyramid

<p align="center">
  <img src="assets/relative_orientation/pyramid_3D.png" width="420">
</p>

### Stairs

<p align="center">
  <img src="assets/relative_orientation/stairs_3D.png" width="420">
</p>

### Topographic map — top view

<p align="center">
  <img src="assets/relative_orientation/map_3D.png" width="420">
</p>

### Topographic map — side view

<p align="center">
  <img src="assets/relative_orientation/side_map_3D.png" width="420">
</p>

---

## 2. Absolute Orientation

Absolute orientation was used as a validation step for the geometric part of the project.

In this case, the 3D coordinates of several calibration points are known. Their corresponding 2D positions are selected in the images, which makes it possible to estimate the camera projection matrices.

The camera projection model is:

$$
\mathbf{x} \sim \mathbf{P}\mathbf{X}.
$$

where:

* $\mathbf{X}$ is a 3D point in homogeneous coordinates;
* $\mathbf{x}$ is its 2D image projection;
* $\mathbf{P}$ is the camera projection matrix.

The projection matrix $\mathbf{P}$ is estimated using DLT from known 3D calibration points and their corresponding 2D image positions.

This part helped validate the core geometric steps:

* Projection
* Camera calibration
* Triangulation
* Reconstruction in a known metric frame

LoFTR was also tested to improve point matching on difficult images, especially when classical local matching was unstable.

---

## Absolute Orientation Error

For the cube experiment, four control points were selected on the top face. Their reconstructed altitude was compared with the theoretical cube height.

The vertical error is:

$$
e_z = z_{\mathrm{rec}} - z_{\mathrm{true}}.
$$

where:

* $z_{\mathrm{rec}}$ is the reconstructed altitude;
* $z_{\mathrm{true}}$ is the theoretical altitude.

The maximum altitude error obtained was about **1.03 mm**.

---

## Technologies

Python, OpenCV, NumPy, SciPy, Matplotlib, SIFT, RANSAC, LoFTR, CloudCompare

---

## Project Context

Academic project at IMT Atlantique on 3D reconstruction by photogrammetry.
