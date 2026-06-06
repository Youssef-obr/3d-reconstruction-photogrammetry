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

$$\mathbf{x}_1 = (u_1, v_1, 1)^{\top}, \qquad \mathbf{x}_2 = (u_2, v_2, 1)^{\top}.$$

The two corresponding points must satisfy the epipolar constraint:

$$\mathbf{x}_2^{\top}\mathbf{F}\mathbf{x}_1 = 0.$$

Here, $\mathbf{F}$ is the fundamental matrix. It describes the projective geometric relation between two camera views.

In practice, some feature matches are wrong. RANSAC is used to robustly estimate $\mathbf{F}$ while rejecting outliers. It repeatedly samples small sets of matches, estimates a candidate fundamental matrix, and keeps the model that explains the largest number of consistent correspondences.

For a candidate fundamental matrix $\mathbf{F}$, the number of inliers can be written as:

$$N_{\mathrm{inliers}}(\mathbf{F}) = \sum_{i=1}^{M} \mathbf{1}\left[d(\mathbf{x}*{2,i}, \mathbf{F}\mathbf{x}*{1,i}) < \tau\right].$$

where:

* $M$ is the number of tentative matches;
* $d$ is the epipolar error;
* $\tau$ is the inlier threshold;
* $\mathbf{1}[\cdot]$ equals 1 if the condition is true and 0 otherwise.

RANSAC keeps the matrix $\mathbf{F}$ with the largest number of inliers. The other matches are rejected as outliers.

Once the fundamental matrix is estimated, the essential matrix is computed using the intrinsic calibration matrix $\mathbf{K}$:

$$\mathbf{E} = \mathbf{K}^{\top}\mathbf{F}\mathbf{K}.$$

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
  <img src="assets/relative_orientation/pyramid_3D.png" width="380">
</p>

### Stairs

<p align="center">
  <img src="assets/relative_orientation/stairs_3D.png" width="380">
</p>

### Topographic map — top view

<p align="center">
  <img src="assets/relative_orientation/map_3D.png" width="380">
</p>

### Topographic map — side view

<p align="center">
  <img src="assets/relative_orientation/side_map_3D.png" width="380">
</p>

---

## 2. Absolute Orientation

Absolute orientation was used as a validation step for the geometric part of the project.

In this case, the 3D coordinates of several calibration points are known. Their corresponding 2D positions are selected in the images, which makes it possible to estimate the camera projection matrices.

The camera projection model is:

$$\mathbf{x} \sim \mathbf{P}\mathbf{X}.$$

where:

* $\mathbf{X}$ is a 3D point in homogeneous coordinates;
* $\mathbf{x}$ is its 2D image projection;
* $\mathbf{P}$ is the camera projection matrix.

The projection matrix $\mathbf{P}$ is estimated using DLT from known 3D calibration points and their corresponding 2D image positions.

---

## Validation on Theoretical Data

Before using automatic feature matching methods such as SIFT or LoFTR, the geometric pipeline was first validated on theoretical data.

The goal was to check that the core photogrammetry algorithm works correctly when the correspondences are known. This validation includes:

* projection of known 3D points into the images;
* DLT calibration of the camera projection matrices;
* extraction of camera parameters;
* triangulation of the reconstructed 3D points;
* comparison with the theoretical object.

The reconstructed shield was compared with the theoretical surface, and the reconstruction error was very low: the average error was about **47.7 µm**, with a maximum error of about **106.5 µm**.

This confirms that the projection, calibration and triangulation pipeline is functional. In later experiments, the main difficulty therefore comes mostly from the quality of image matching, not from the geometric model itself.

<p align="center">
  <img src="assets/absolute_orientation/shield_validation.png" width="520">
</p>

---

## Absolute Orientation on Real Images

After validating the theoretical inverse problem, the same geometric principles were applied to real images with calibration points.

LoFTR was tested to improve point matching on difficult images, especially when classical local matching was unstable.

For the cube experiment, four control points were selected on the top face. Their reconstructed altitude was compared with the theoretical cube height.

The vertical error is:

$$e_z = z_{\mathrm{rec}} - z_{\mathrm{true}}.$$

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
