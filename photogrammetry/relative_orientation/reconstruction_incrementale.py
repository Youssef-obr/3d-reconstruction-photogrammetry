import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_FOLDER = "images_multivues"
OUTPUT_FOLDER = "resultats_incremental"

RATIO_TEST = 0.75
RANSAC_THRESH = 1.0
MIN_INLIERS_SAVE = 100
MIN_PNP_CORR = 6

FX0 = 3028.0
FY0 = 3028.0
REF_W = 3024.0
REF_H = 4032.0


def load_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []

    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    paths = sorted(set(os.path.abspath(p) for p in paths), key=os.path.getmtime)

    if len(paths) < 2:
        raise RuntimeError("Il faut au moins deux images dans images_multivues.")

    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Image illisible : {p}")
        images.append(img)

    return paths, images


def iphone13_K(w, h):
    fx = FX0 * (w / REF_W)
    fy = FY0 * (h / REF_H)
    cx = w / 2.0
    cy = h / 2.0

    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def create_sift():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        raise RuntimeError("SIFT non disponible. Installe opencv-contrib-python.")


def detect_sift_all(images):
    sift = create_sift()
    keypoints, descriptors = [], []

    for i, img in enumerate(images):
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
        print(f"{i:03d} | keypoints = {len(kp)}")

    return keypoints, descriptors


def match_pair(kp_i, kp_j, des_i, des_j, ratio=RATIO_TEST, ransac_thresh=RANSAC_THRESH):
    if des_i is None or des_j is None:
        return {
            "good": [],
            "inliers": [],
            "pts_i": None,
            "pts_j": None,
            "F": None,
            "mask": None,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_i, des_j, k=2)

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 8:
        return {
            "good": good,
            "inliers": [],
            "pts_i": None,
            "pts_j": None,
            "F": None,
            "mask": None,
        }

    pts_i_all = np.float32([kp_i[m.queryIdx].pt for m in good])
    pts_j_all = np.float32([kp_j[m.trainIdx].pt for m in good])

    try:
        F, mask = cv2.findFundamentalMat(
            pts_i_all,
            pts_j_all,
            cv2.FM_RANSAC,
            ransac_thresh
        )
    except cv2.error:
        F, mask = None, None

    if F is None or mask is None:
        return {
            "good": good,
            "inliers": [],
            "pts_i": None,
            "pts_j": None,
            "F": None,
            "mask": None,
        }

    mask = mask.ravel().astype(bool)
    inliers = [m for m, keep in zip(good, mask) if keep]

    return {
        "good": good,
        "inliers": inliers,
        "pts_i": pts_i_all[mask],
        "pts_j": pts_j_all[mask],
        "F": F,
        "mask": mask,
    }


def save_matches(img_i, img_j, kp_i, kp_j, matches, output_path):
    img_matches = cv2.drawMatches(
        img_i, kp_i,
        img_j, kp_j,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, img_matches)


def match_all_pairs(images, keypoints, descriptors, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pairs = {}
    n = len(images)

    for i in range(n):
        for j in range(i + 1, n):
            res = match_pair(keypoints[i], keypoints[j], descriptors[i], descriptors[j])
            pairs[(i, j)] = res

            n_good = len(res["good"])
            n_in = len(res["inliers"])

            print(f"paire {i:03d}-{j:03d} | good = {n_good} | inliers = {n_in}")

            if n_in >= MIN_INLIERS_SAVE:
                out = os.path.join(output_folder, f"matches_{i:03d}_{j:03d}.png")
                save_matches(
                    images[i],
                    images[j],
                    keypoints[i],
                    keypoints[j],
                    res["inliers"],
                    out
                )

    return pairs


def get_pair_result(pairs, a, b):
    if (a, b) in pairs:
        return pairs[(a, b)], False
    if (b, a) in pairs:
        return pairs[(b, a)], True
    return None, False


def inlier_keypoint_pairs(pair_res, reverse=False):
    q = [m.queryIdx for m in pair_res["inliers"]]
    t = [m.trainIdx for m in pair_res["inliers"]]

    if reverse:
        return list(zip(t, q))

    return list(zip(q, t))


def projection_matrix(K, R, t):
    return K @ np.hstack([R, np.asarray(t, dtype=float).reshape(3, 1)])


def triangulate_points(P0, P1, pts0, pts1):
    Xh = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    X = (Xh[:3] / Xh[3]).T
    return X


def filter_triangulated_points(X, R0, t0, R1, t1):
    finite = np.isfinite(X).all(axis=1)

    t0 = np.asarray(t0, dtype=float).reshape(3, 1)
    t1 = np.asarray(t1, dtype=float).reshape(3, 1)

    X0 = (R0 @ X.T + t0).T
    X1 = (R1 @ X.T + t1).T

    positive = (X0[:, 2] > 0) & (X1[:, 2] > 0)

    return finite & positive


def add_points_from_pair(
    i,
    j,
    pair_res,
    reverse,
    K,
    R_global,
    t_global,
    points_3d,
    obs_to_point,
    keypoints
):
    kp_pairs = inlier_keypoint_pairs(pair_res, reverse)

    if len(kp_pairs) < 8:
        return 0

    pts_i, pts_j, kept_pairs = [], [], []

    for kpi, kpj in kp_pairs:
        if (i, kpi) in obs_to_point and (j, kpj) in obs_to_point:
            continue

        pts_i.append(keypoints[i][kpi].pt)
        pts_j.append(keypoints[j][kpj].pt)
        kept_pairs.append((kpi, kpj))

    if len(pts_i) < 8:
        return 0

    pts_i = np.float32(pts_i)
    pts_j = np.float32(pts_j)

    P_i = projection_matrix(K, R_global[i], t_global[i])
    P_j = projection_matrix(K, R_global[j], t_global[j])

    X = triangulate_points(P_i, P_j, pts_i, pts_j)
    valid = filter_triangulated_points(
        X,
        R_global[i],
        t_global[i],
        R_global[j],
        t_global[j]
    )

    added = 0

    for Xp, ok, (kpi, kpj) in zip(X, valid, kept_pairs):
        if not ok:
            continue

        idx = len(points_3d)
        points_3d.append(Xp)

        obs_to_point[(i, kpi)] = idx
        obs_to_point[(j, kpj)] = idx

        added += 1

    return added


def find_2d_3d_correspondences(img_idx, registered, pairs, obs_to_point, points_3d, keypoints):
    pts2d, pts3d = [], []
    used_points = set()

    for r in registered:
        pair_res, reverse = get_pair_result(pairs, img_idx, r)

        if pair_res is None:
            continue

        kp_pairs = inlier_keypoint_pairs(pair_res, reverse)

        for kp_img, kp_reg in kp_pairs:
            key = (r, kp_reg)

            if key not in obs_to_point:
                continue

            pid = obs_to_point[key]

            if pid in used_points:
                continue

            pts2d.append(keypoints[img_idx][kp_img].pt)
            pts3d.append(points_3d[pid])
            used_points.add(pid)

    if len(pts2d) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    return np.float32(pts2d), np.float32(pts3d)


def set_axes_equal(ax, points):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        return

    mn = points.min(axis=0)
    mx = points.max(axis=0)

    center = 0.5 * (mn + mx)
    radius = 0.5 * np.max(mx - mn)

    if radius <= 0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def get_camera_centers(R_global, t_global, registered):
    centers = []

    for i in registered:
        if R_global[i] is None or t_global[i] is None:
            continue

        R = np.asarray(R_global[i], dtype=float)
        t = np.asarray(t_global[i], dtype=float).reshape(3, 1)
        C = -R.T @ t

        centers.append((i, C.ravel()))

    return centers


def plot_3d_improved(points_3d, R_global, t_global, registered, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    X = np.asarray(points_3d, dtype=float)
    centers_data = get_camera_centers(R_global, t_global, registered)
    centers = np.asarray([C for _, C in centers_data], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(X) > 0:
        sc = ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            c=X[:, 2],
            s=3,
            marker=".",
            cmap="viridis",
            label="Points 3D"
        )
        fig.colorbar(sc, ax=ax, shrink=0.65, label="Z")

    if len(centers) > 0:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="red",
            s=70,
            marker="^",
            label="Caméras"
        )

        ax.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c="red",
            linewidth=1.5,
            label="Trajectoire"
        )

        for idx, C in centers_data:
            ax.text(C[0], C[1], C[2], str(idx), color="black")

    if len(X) > 0 and len(centers) > 0:
        all_pts = np.vstack([X, centers])
    elif len(X) > 0:
        all_pts = X
    else:
        all_pts = centers

    set_axes_equal(ax, all_pts)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reconstruction incrémentale - vue 3D améliorée")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "vue_3d_amelioree.png"), dpi=250)
    plt.show()


def plot_2d_projections(points_3d, R_global, t_global, registered, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    X = np.asarray(points_3d, dtype=float)
    centers_data = get_camera_centers(R_global, t_global, registered)
    centers = np.asarray([C for _, C in centers_data], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    views = [
        (0, 1, "X", "Y", "Projection XY"),
        (0, 2, "X", "Z", "Projection XZ"),
        (1, 2, "Y", "Z", "Projection YZ"),
    ]

    for ax, (a, b, la, lb, title) in zip(axes, views):
        if len(X) > 0:
            ax.scatter(
                X[:, a],
                X[:, b],
                c=X[:, 2],
                s=3,
                marker=".",
                cmap="viridis"
            )

        if len(centers) > 0:
            ax.scatter(
                centers[:, a],
                centers[:, b],
                c="red",
                s=45,
                marker="^"
            )

            ax.plot(
                centers[:, a],
                centers[:, b],
                c="red",
                linewidth=1.2
            )

            for idx, C in centers_data:
                ax.text(C[a], C[b], str(idx), color="black")

        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "projections_2d.png"), dpi=250)
    plt.show()


"""def export_ply(points_3d, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X = np.asarray(points_3d, dtype=float)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in X:
            f.write(f"{x} {y} {z}\n")

    print("PLY sauvegardé :", output_path)"""


def visualize_open3d(ply_path):
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D non installé. Visualisation interactive ignorée.")
        return

    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])


import os
import numpy as np
import matplotlib.pyplot as plt


def clean_and_center_points(points_3d, percentile=95):
    X = np.asarray(points_3d, dtype=float)
    n_before = len(X)

    X = X[np.isfinite(X).all(axis=1)]

    if len(X) == 0:
        barycentre = np.zeros(3)
        print("Nettoyage nuage 3D")
        print("Points avant nettoyage :", n_before)
        print("Points après nettoyage :", 0)
        print("Barycentre utilisé :", barycentre)
        print("Percentile utilisé :", percentile)
        return X, barycentre

    barycentre = np.median(X, axis=0)
    X_centered = X - barycentre

    dist = np.linalg.norm(X_centered, axis=1)
    seuil = np.percentile(dist, percentile)

    X_clean = X_centered[dist <= seuil]

    print("Nettoyage nuage 3D")
    print("Points avant nettoyage :", n_before)
    print("Points après nettoyage :", len(X_clean))
    print("Barycentre utilisé :", barycentre)
    print("Percentile utilisé :", percentile)

    return X_clean, barycentre


def export_ply(points_3d, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X = np.asarray(points_3d, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(X)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in X:
            f.write(f"{x} {y} {z}\n")

    print("PLY sauvegardé :", output_path)


def set_axes_equal(ax, points):
    X = np.asarray(points, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]

    if len(X) == 0:
        return

    mn = X.min(axis=0)
    mx = X.max(axis=0)
    center = 0.5 * (mn + mx)
    radius = 0.5 * np.max(mx - mn)

    if radius <= 0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_clean_centered_cloud(
    X_clean,
    output_path="resultats_incremental/nuage_clean_centered.png"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X = np.asarray(X_clean, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(X) > 0:
        sc = ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=X[:, 2],
            s=3,
            marker=".",
            cmap="viridis"
        )
        fig.colorbar(sc, ax=ax, shrink=0.65, label="Z")

    set_axes_equal(ax, X)

    ax.set_xlabel("X centré")
    ax.set_ylabel("Y centré")
    ax.set_zlabel("Z centré")
    ax.set_title("Nuage 3D nettoyé et centré")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.show()

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    paths, images = load_images(IMAGE_FOLDER)

    print("Images chargées :")
    for i, p in enumerate(paths):
        print(f"{i:03d} | {os.path.basename(p)}")
    print()

    h, w = images[0].shape[:2]
    K = iphone13_K(w, h)

    print("K estimé :")
    print(K)
    print()

    print("Détection SIFT :")
    keypoints, descriptors = detect_sift_all(images)
    print()

    print("Matching exhaustif :")
    pairs = match_all_pairs(images, keypoints, descriptors, OUTPUT_FOLDER)
    print()

    valid_pairs = [
        (ij, res)
        for ij, res in pairs.items()
        if res["F"] is not None and len(res["inliers"]) >= 8
    ]

    if not valid_pairs:
        raise RuntimeError("Aucune paire valide pour initialiser la reconstruction.")

    (i0, j0), best = max(valid_pairs, key=lambda item: len(item[1]["inliers"]))

    print("Meilleure paire initiale :")
    print(f"{i0:03d}-{j0:03d} | inliers = {len(best['inliers'])}")
    print()

    E = K.T @ best["F"] @ K
    _, R_rel, t_rel, _ = cv2.recoverPose(E, best["pts_i"], best["pts_j"], K)

    n = len(images)

    R_global = [None] * n
    t_global = [None] * n

    R_global[i0] = np.eye(3)
    t_global[i0] = np.zeros(3)

    R_global[j0] = R_rel
    t_global[j0] = t_rel.reshape(3)

    registered = [i0, j0]
    points_3d = []
    obs_to_point = {}

    added_initial = add_points_from_pair(
        i0,
        j0,
        best,
        False,
        K,
        R_global,
        t_global,
        points_3d,
        obs_to_point,
        keypoints
    )

    print("Points triangulés initialement :", added_initial)
    print()

    remaining = set(range(n)) - set(registered)

    while remaining:
        best_candidate = None

        for img_idx in list(remaining):
            pts2d, pts3d = find_2d_3d_correspondences(
                img_idx,
                registered,
                pairs,
                obs_to_point,
                points_3d,
                keypoints
            )

            if len(pts2d) >= MIN_PNP_CORR:
                if best_candidate is None or len(pts2d) > best_candidate[0]:
                    best_candidate = (len(pts2d), img_idx, pts2d, pts3d)

        if best_candidate is None:
            print("Aucune image restante avec assez de correspondances 2D-3D.")
            break

        n_corr, img_idx, pts2d, pts3d = best_candidate

        print(f"Image {img_idx:03d} | correspondances 2D-3D = {n_corr}", end=" | ")

        try:
            ok, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                K,
                None,
                iterationsCount=1000,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except cv2.error:
            ok, rvec, tvec, pnp_inliers = False, None, None, None

        if not ok or pnp_inliers is None or len(pnp_inliers) < MIN_PNP_CORR:
            print("PnP échec")
            remaining.remove(img_idx)
            continue

        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.reshape(3)

        R_global[img_idx] = R_new
        t_global[img_idx] = t_new

        registered.append(img_idx)
        remaining.remove(img_idx)

        new_points = 0

        for r in registered:
            if r == img_idx:
                continue

            pair_res, reverse = get_pair_result(pairs, img_idx, r)

            if pair_res is None:
                continue

            new_points += add_points_from_pair(
                img_idx,
                r,
                pair_res,
                reverse,
                K,
                R_global,
                t_global,
                points_3d,
                obs_to_point,
                keypoints
            )

        print(f"PnP succès | inliers PnP = {len(pnp_inliers)} | nouveaux points = {new_points}")

    print()
    print("Nombre final de caméras intégrées :", len(registered), "/", n)
    print("Nombre final de points 3D :", len(points_3d))
    print("Pas de bundle adjustment, dérive possible, échelle arbitraire.")
    print("Pas de surface/mesh : nuage sparse uniquement.")
    print()

    plot_3d_improved(
        points_3d,
        R_global,
        t_global,
        registered,
        OUTPUT_FOLDER
    )

    plot_2d_projections(
        points_3d,
        R_global,
        t_global,
        registered,
        OUTPUT_FOLDER
    )

    export_ply(
        points_3d,
        os.path.join(OUTPUT_FOLDER, "nuage_incremental_brut.ply")
    )

    X_clean_centered, barycentre = clean_and_center_points(
        points_3d,
        percentile=95
    )

    export_ply(
        X_clean_centered,
        os.path.join(OUTPUT_FOLDER, "nuage_incremental_clean_centered.ply")
    )

    plot_clean_centered_cloud(
        X_clean_centered,
        output_path=os.path.join(OUTPUT_FOLDER, "nuage_clean_centered.png")
    )


if __name__ == "__main__":
    main()