import itertools
from dataclasses import dataclass
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import rq



# Donnees et structures

@dataclass
class CameraSpec:
    delta_mm: float
    f_mm: float
    nh_pix: int
    nv_pix: int
    xh_mm: float
    yh_mm: float


@dataclass
class CameraPoseInput:
    theta_x_deg: float
    theta_y_deg: float
    theta_z_deg: float
    x_tilde_o_mm: np.ndarray  # coordonnees de O dans le repere camera



# Outils mathematiques

def deg2rad(angle_deg: float) -> float:
    return np.deg2rad(angle_deg)



def rot_x(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )



def rot_y(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )



def rot_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )



def rotation_from_angles(theta_x_deg: float, theta_y_deg: float, theta_z_deg: float) -> np.ndarray:
    tx = deg2rad(theta_x_deg)
    ty = deg2rad(theta_y_deg)
    tz = deg2rad(theta_z_deg)
    # Equation (7): R = Rx * Ry * Rz
    return rot_x(tx) @ rot_y(ty) @ rot_z(tz)



def build_k_matrix(f_mm: float, xh_mm: float, yh_mm: float) -> np.ndarray:
    # Equation (4)
    return np.array(
        [
            [-f_mm, 0.0, xh_mm],
            [0.0, -f_mm, yh_mm],
            [0.0, 0.0, 1.0],
        ]
    )



def build_n_matrix(delta_mm: float) -> np.ndarray:
    # Equation (5)
    return np.diag([1.0, 1.0, delta_mm])



def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )



def homogenize_3d(points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1))
    return np.hstack([points_xyz, ones])



def build_projection_matrix(spec: CameraSpec, pose: CameraPoseInput):
    """
    P = N K [R | -R X_omega]
    avec X_omega = -R^{-1} x_tilde_O = -R^T x_tilde_O.
    """
    r = rotation_from_angles(pose.theta_x_deg, pose.theta_y_deg, pose.theta_z_deg)
    x_omega = -r.T @ pose.x_tilde_o_mm.reshape(3, 1)

    k = build_k_matrix(spec.f_mm, spec.xh_mm, spec.yh_mm)
    n = build_n_matrix(spec.delta_mm)
    p = n @ k @ np.hstack([r, -r @ x_omega])
    return p, r, x_omega



def project_points(p: np.ndarray, points_xyz: np.ndarray):
    xh = homogenize_3d(points_xyz)
    nh = (p @ xh.T).T
    p_pix = nh[:, 0] / nh[:, 2]
    q_pix = nh[:, 1] / nh[:, 2]
    return p_pix, q_pix



def flatten_valid_grid(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray):
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    valid = ~(np.isnan(x_flat) | np.isnan(y_flat) | np.isnan(z_flat))
    xyz = np.column_stack([x_flat[valid], y_flat[valid], z_flat[valid]])
    return xyz, valid


# DLT calibration

def build_dlt_matrix(points_xyz: np.ndarray, p_pix: np.ndarray, q_pix: np.ndarray) -> np.ndarray:
    rows = []
    for i in range(points_xyz.shape[0]):
        x = np.array([points_xyz[i, 0], points_xyz[i, 1], points_xyz[i, 2], 1.0])
        n = np.array([p_pix[i], q_pix[i], 1.0])
        c3 = np.kron(skew(n), x)
        rows.append(c3[0, :])
        rows.append(c3[1, :])
    return np.vstack(rows)



def solve_dlt(points_xyz: np.ndarray, p_pix: np.ndarray, q_pix: np.ndarray) -> np.ndarray:
    c = build_dlt_matrix(points_xyz, p_pix, q_pix)
    _, _, vt = np.linalg.svd(c)
    p_vec = vt[-1, :]
    return p_vec.reshape(3, 4)



def normalize_projection_frobenius(p: np.ndarray) -> np.ndarray:
    return p / np.linalg.norm(p)



def max_relative_error_percent(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    mask = np.abs(b) > eps
    rel = np.abs(a[mask] / b[mask] - 1.0)
    return 100.0 * np.max(rel)



def align_scale_to_reference(p_est: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    alpha = np.sum(p_est * p_ref) / np.sum(p_est * p_est)
    return alpha * p_est


# Extraction des parametres de projection

def extract_angles_from_rotation(r: np.ndarray):
    """
    Extraction des angles pour R = Rx(theta_x) Ry(theta_y) Rz(theta_z)
    """
    theta_y = np.arcsin(np.clip(r[0, 2], -1.0, 1.0))
    cy = np.cos(theta_y)
    if abs(cy) > 1e-10:
        theta_x = np.arctan2(-r[1, 2], r[2, 2])
        theta_z = np.arctan2(-r[0, 1], r[0, 0])
    else:
        theta_x = 0.0
        theta_z = np.arctan2(r[1, 0], r[1, 1])
    return np.rad2deg(theta_x), np.rad2deg(theta_y), np.rad2deg(theta_z)



def extract_parameters_from_projection(p: np.ndarray, delta_known_mm: float):
    """
    Decomposition de Ps = P[:, :3] sous la forme Ps = (N K) R.
    La decomposition RQ n'est pas unique a des signes pres.
    On selectionne la branche physiquement valide :
    - delta > 0
    - f > 0
    - xH > 0, yH > 0
    - z_O(camera) < 0
    """
    ps = p[:, :3]
    pv = p[:, 3].reshape(3, 1)
    x_omega_object = -np.linalg.inv(ps) @ pv

    u0, q0 = rq(ps)
    best = None

    for signs in itertools.product([-1.0, 1.0], repeat=3):
        if np.prod(signs) < 0:
            continue
        s = np.diag(signs)
        u = u0 @ s
        r = s @ q0

        if abs(u[2, 2]) < 1e-20:
            continue

        lam = u[2, 2] / delta_known_mm
        if abs(lam) < 1e-20:
            continue

        u_metric = u / lam
        delta_mm = u_metric[2, 2]
        f_mm = -0.5 * (u_metric[0, 0] + u_metric[1, 1])
        xh_mm = u_metric[0, 2]
        yh_mm = u_metric[1, 2]
        x_tilde_o_mm = (-r @ x_omega_object).flatten()
        theta_x_deg, theta_y_deg, theta_z_deg = extract_angles_from_rotation(r)

        penalty = 0.0
        penalty += 1000.0 * abs(delta_mm - delta_known_mm)
        if f_mm <= 0:
            penalty += 100.0
        if xh_mm <= 0:
            penalty += 10.0
        if yh_mm <= 0:
            penalty += 10.0
        if x_tilde_o_mm[2] >= 0:
            penalty += 20.0
        if abs(theta_x_deg) > 90.0:
            penalty += 5.0
        if abs(theta_y_deg) > 90.0:
            penalty += 5.0

        candidate = {
            "penalty": penalty,
            "delta_mm": delta_mm,
            "f_mm": f_mm,
            "xh_mm": xh_mm,
            "yh_mm": yh_mm,
            "x_omega_object_mm": x_omega_object.flatten(),
            "x_tilde_o_mm": x_tilde_o_mm,
            "theta_x_deg": theta_x_deg,
            "theta_y_deg": theta_y_deg,
            "theta_z_deg": theta_z_deg,
            "u_metric": u_metric,
            "r_matrix": r,
        }

        if best is None or candidate["penalty"] < best["penalty"]:
            best = candidate

    return best


# Mesure 3D 

def build_measurement_matrix(p_list, p_obs, q_obs) -> np.ndarray:
    rows = []
    for p, pp, qq in zip(p_list, p_obs, q_obs):
        n = np.array([pp, qq, 1.0])
        mi = skew(n) @ p
        rows.append(mi[0, :])
        rows.append(mi[1, :])
    return np.vstack(rows)



def triangulate_point(p_list, p_obs, q_obs):
    m = build_measurement_matrix(p_list, p_obs, q_obs)
    _, _, vt = np.linalg.svd(m)
    xh = vt[-1, :]
    xh = xh / xh[3]
    return xh[:3], m


# =============================================================================
# Donnees PDF
# =============================================================================



def load_pdf_data():
    delta_mm = 1.22e-3
    f_mm = 3.99
    nh = 4032
    nv = 3024
    xh_mm = (nv / 2.0) * delta_mm
    yh_mm = (nh / 2.0) * delta_mm

    spec = CameraSpec(
        delta_mm=delta_mm,
        f_mm=f_mm,
        nh_pix=nh,
        nv_pix=nv,
        xh_mm=xh_mm,
        yh_mm=yh_mm,
    )

    # IMPORTANT : ordre des lignes conforme aux tableaux p1,q1,p2,q2,p3,q3 du PDF
    x_cols = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80], dtype=float)
    y_rows = np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80], dtype=float)

    x_th = np.tile(x_cols, (9, 1))
    y_th = np.tile(y_rows.reshape(-1, 1), (1, 9))
    z_th = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2.1029, 7.8193, 9.6586, 7.8193, 2.1029, 0, 0],
            [0, 2.1029, 11.4675, 16.7262, 18.4272, 16.7262, 11.4675, 2.1029, 0],
            [0, 7.8193, 16.7262, 21.7580, 23.3896, 21.7580, 16.7262, 7.8193, 0],
            [0, 9.6586, 18.4272, 23.3896, 25.0000, 23.3896, 18.4272, 9.6586, 0],
            [0, 7.8193, 16.7262, 21.7580, 23.3896, 21.7580, 16.7262, 7.8193, 0],
            [0, 2.1029, 11.4675, 16.7262, 18.4272, 16.7262, 11.4675, 2.1029, 0],
            [0, 0, 2.1029, 7.8193, 9.6586, 7.8193, 2.1029, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    xc = np.array([[-100, 0, 100], [-100, np.nan, 100], [-100, 0, 100]], dtype=float)
    yc = np.array([[100, 100, 100], [0, np.nan, 0], [-100, -100, -100]], dtype=float)
    zc = np.array([[100, 0, 100], [0, np.nan, 0], [100, 0, 100]], dtype=float)

    p_c1 = np.array([[872, 2330, 2524], [1254, np.nan, 2330], [140, 1254, 872]], dtype=float)
    q_c1 = np.array([[3800, 2842, 2016], [2625, np.nan, 1190], [2016, 1407, 232]], dtype=float)

    p_c2 = np.array([[133, 1077, 843], [1319, np.nan, 2112], [1533, 2463, 2855]], dtype=float)
    q_c2 = np.array([[1769, 2491, 3605], [1382, np.nan, 2932], [454, 1616, 2725]], dtype=float)

    p_c3 = np.array([[166, 1716, 2521], [923, np.nan, 2803], [166, 1716, 2521]], dtype=float)
    q_c3 = np.array([[3170, 3038, 3799], [2016, np.nan, 2016], [862, 994, 233]], dtype=float)

    p1 = np.array(
        [
            [1033, 1103, 1177, 1254, 1335, 1421, 1512, 1608, 1710],
            [1103, 1177, 1240, 1283, 1356, 1460, 1594, 1710, 1819],
            [1177, 1240, 1258, 1307, 1386, 1494, 1633, 1805, 1934],
            [1254, 1283, 1307, 1362, 1446, 1560, 1705, 1882, 2057],
            [1335, 1356, 1386, 1446, 1536, 1657, 1809, 1994, 2189],
            [1421, 1460, 1494, 1560, 1657, 1785, 1945, 2138, 2330],
            [1512, 1594, 1633, 1705, 1809, 1945, 2114, 2317, 2481],
            [1608, 1710, 1805, 1882, 1994, 2138, 2317, 2481, 2644],
            [1710, 1819, 1934, 2057, 2189, 2330, 2481, 2644, 2820],
        ],
        dtype=float,
    )
    q1 = np.array(
        [
            [2016, 1900, 1779, 1651, 1516, 1374, 1223, 1064, 895],
            [2132, 2016, 1894, 1762, 1623, 1479, 1333, 1175, 1004],
            [2253, 2138, 2016, 1883, 1741, 1593, 1441, 1290, 1121],
            [2381, 2270, 2149, 2016, 1873, 1722, 1566, 1408, 1245],
            [2516, 2409, 2291, 2159, 2016, 1864, 1705, 1543, 1378],
            [2658, 2553, 2439, 2310, 2168, 2016, 1855, 1691, 1520],
            [2809, 2699, 2591, 2466, 2327, 2177, 2016, 1850, 1673],
            [2968, 2857, 2742, 2624, 2489, 2341, 2182, 2016, 1838],
            [3137, 3028, 2911, 2787, 2654, 2512, 2359, 2194, 2016],
        ],
        dtype=float,
    )

    p2 = np.array(
        [
            [1929, 2016, 2106, 2197, 2292, 2390, 2490, 2594, 2701],
            [1785, 1867, 1945, 2013, 2098, 2199, 2312, 2416, 2518],
            [1647, 1717, 1762, 1826, 1908, 2006, 2117, 2242, 2344],
            [1515, 1557, 1597, 1658, 1735, 1830, 1939, 2062, 2179],
            [1389, 1418, 1453, 1509, 1582, 1673, 1779, 1900, 2023],
            [1269, 1302, 1331, 1382, 1451, 1538, 1641, 1760, 1873],
            [1154, 1210, 1233, 1278, 1343, 1426, 1526, 1644, 1731],
            [1043, 1106, 1161, 1200, 1260, 1340, 1438, 1520, 1596],
            [937, 998, 1060, 1124, 1189, 1256, 1324, 1394, 1466],
        ],
        dtype=float,
    )
    q2 = np.array(
        [
            [1102, 1251, 1404, 1561, 1723, 1890, 2063, 2240, 2423],
            [1214, 1360, 1509, 1659, 1821, 1990, 2160, 2334, 2513],
            [1320, 1462, 1602, 1756, 1919, 2088, 2258, 2425, 2599],
            [1422, 1556, 1699, 1853, 2016, 2184, 2352, 2516, 2680],
            [1519, 1652, 1795, 1949, 2110, 2275, 2440, 2601, 2757],
            [1612, 1746, 1888, 2040, 2199, 2360, 2521, 2675, 2830],
            [1701, 1837, 1977, 2126, 2280, 2437, 2591, 2739, 2900],
            [1786, 1921, 2059, 2204, 2353, 2504, 2651, 2805, 2967],
            [1867, 2001, 2137, 2277, 2420, 2567, 2717, 2872, 3030],
        ],
        dtype=float,
    )

    p3 = np.array(
        [
            [1064, 1213, 1371, 1539, 1716, 1906, 2108, 2324, 2555],
            [1064, 1213, 1360, 1500, 1671, 1872, 2100, 2324, 2555],
            [1064, 1202, 1310, 1453, 1628, 1832, 2063, 2317, 2555],
            [1064, 1170, 1281, 1426, 1603, 1809, 2042, 2298, 2555],
            [1064, 1160, 1271, 1417, 1594, 1801, 2035, 2292, 2555],
            [1064, 1170, 1281, 1426, 1603, 1809, 2042, 2298, 2555],
            [1064, 1202, 1310, 1453, 1628, 1832, 2063, 2317, 2555],
            [1064, 1213, 1360, 1500, 1671, 1872, 2100, 2324, 2555],
            [1064, 1213, 1371, 1539, 1716, 1906, 2108, 2324, 2555],
        ],
        dtype=float,
    )
    q3 = np.array(
        [
            [1289, 1268, 1246, 1223, 1198, 1172, 1144, 1114, 1082],
            [1471, 1455, 1436, 1409, 1386, 1369, 1358, 1339, 1315],
            [1653, 1640, 1620, 1601, 1586, 1573, 1565, 1562, 1549],
            [1834, 1825, 1815, 1806, 1798, 1791, 1787, 1785, 1782],
            [2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016],
            [2198, 2207, 2217, 2226, 2234, 2241, 2245, 2247, 2250],
            [2379, 2392, 2412, 2431, 2446, 2459, 2467, 2470, 2483],
            [2561, 2577, 2596, 2623, 2646, 2663, 2674, 2693, 2717],
            [2743, 2764, 2786, 2809, 2834, 2860, 2888, 2918, 2950],
        ],
        dtype=float,
    )

    p1_norm_pdf = np.array(
        [
            [1.0081, 1.0081, -3.8772, 646.9523],
            [-3.8067, 1.4961, -1.6330, 762.5065],
            [-0.0006, -0.0006, -0.0008, 0.3782],
        ],
        dtype=float,
    )
    p2_norm_pdf = np.array(
        [
            [-1.0397, 1.7999, 2.9499, -625.8119],
            [-2.3317, -2.5131, 1.7499, -779.9561],
            [0.0003, -0.0004, 0.0009, -0.3706],
        ],
        dtype=float,
    )
    p3_norm_pdf = np.array(
        [
            [-2.4512, -0.0000, 3.4752, -648.1783],
            [1.1896, -3.8602, 2.0605, -761.4632],
            [0.0006, -0.0000, 0.0010, -0.3777],
        ],
        dtype=float,
    )

    max_errors_pdf = [0.0609, 0.1259, 0.0256]
    
    poses = [
        CameraPoseInput(0.0, -45.0, -45.0, np.array([20.0, 0.0, -330.0])),
        CameraPoseInput(0.0, -30.0, 60.0, np.array([20.0, 10.0, -370.0])),
        CameraPoseInput(0.0, -30.0, 0.0, np.array([20.0, 0.0, -320.0])),
    ]

    return {
        "spec": spec,
        "x_th": x_th,
        "y_th": y_th,
        "z_th": z_th,
        "xc": xc,
        "yc": yc,
        "zc": zc,
        "p_c": [p_c1, p_c2, p_c3],
        "q_c": [q_c1, q_c2, q_c3],
        "p": [p1, p2, p3],
        "q": [q1, q2, q3],
        "p_norm_pdf": [p1_norm_pdf, p2_norm_pdf, p3_norm_pdf],
        "max_errors_pdf": max_errors_pdf,
        "poses": poses,
    }


# Validation par blocs

def print_matrix(name: str, m: np.ndarray):
    print(f"{name} =")
    print(np.array2string(m, precision=4, suppress_small=False))
    print()



def validate_direct_problem(data):
    spec = data["spec"]
    x_measure = np.column_stack([data["x_th"].flatten(), data["y_th"].flatten(), data["z_th"].flatten()])

    p_th_list = []
    r_list = []
    x_omega_list = []

    print("=== Script Images.m : problème direct ===")
    for idx, pose in enumerate(data["poses"]):
        p_th, r, x_omega = build_projection_matrix(spec, pose)
        p_th_list.append(p_th)
        r_list.append(r)
        x_omega_list.append(x_omega)

        p_calc, q_calc = project_points(p_th, x_measure)
        p_calc = np.round(p_calc).reshape(9, 9)
        q_calc = np.round(q_calc).reshape(9, 9)

        err_p = np.max(np.abs(p_calc - data["p"][idx]))
        err_q = np.max(np.abs(q_calc - data["q"][idx]))
        err_pc = np.max(np.abs(np.round(project_points(p_th, flatten_valid_grid(data["xc"], data["yc"], data["zc"])[0])[0]) - data["p_c"][idx].flatten()[~np.isnan(data["p_c"][idx].flatten())]))
        err_qc = np.max(np.abs(np.round(project_points(p_th, flatten_valid_grid(data["xc"], data["yc"], data["zc"])[0])[1]) - data["q_c"][idx].flatten()[~np.isnan(data["q_c"][idx].flatten())]))

        print(f"Image {idx + 1}")
        print(f"  theta_x, theta_y, theta_z [deg] = ({pose.theta_x_deg:.1f}, {pose.theta_y_deg:.1f}, {pose.theta_z_deg:.1f})")
        print(f"  x_tilde_O [mm] = {pose.x_tilde_o_mm}")
        print(f"  max|pC_calc - pC_pdf| = {err_pc:.0f} pix")
        print(f"  max|qC_calc - qC_pdf| = {err_qc:.0f} pix")
        print(f"  max|p_calc - p_pdf|   = {err_p:.0f} pix")
        print(f"  max|q_calc - q_pdf|   = {err_q:.0f} pix")
        print()

    return {
        "p_th_list": p_th_list,
        "r_list": r_list,
        "x_omega_list": x_omega_list,
        "x_measure": x_measure,
    }



def validate_dlt_calibration(data, direct_results):
    calib_xyz, calib_valid = flatten_valid_grid(data["xc"], data["yc"], data["zc"])

    p_est_list = []
    p_est_norm_list = []
    p_pdf_display_calc = []
    max_errors = []

    print("=== Script Reconstruction3Dsimul.m : calibration DLT ===")
    for idx in range(3):
        p_c_flat = data["p_c"][idx].flatten()[calib_valid]
        q_c_flat = data["q_c"][idx].flatten()[calib_valid]

        p_est = solve_dlt(calib_xyz, p_c_flat, q_c_flat)
        p_est_list.append(p_est)

        p_est_norm = normalize_projection_frobenius(p_est)
        p_est_norm_list.append(p_est_norm)

        p_th = direct_results["p_th_list"][idx]
        alpha_err = np.sum(p_est * p_th) / np.sum(p_est * p_est)
        p_est_for_err = alpha_err * p_est
        max_err = max_relative_error_percent(p_est_for_err, p_th)
        max_errors.append(max_err)

        p_display = align_scale_to_reference(p_est, data["p_norm_pdf"][idx])
        p_pdf_display_calc.append(p_display)

        print(f"Image {idx + 1}")
        print_matrix(f"P{idx + 1}.norm calcule (format PDF)", p_display)
        print_matrix(f"P{idx + 1}.norm PDF", data["p_norm_pdf"][idx])
        print(
            f"Max(|P{idx + 1}.norm/P{idx + 1}.th - 1|) = {max_err:.4f}%  "
            f"(PDF: {data['max_errors_pdf'][idx]:.4f}%)"
        )
        print(
            f"Max abs diff matrice vs PDF = {np.max(np.abs(p_display - data['p_norm_pdf'][idx])):.6f}"
        )
        print()

    return {
        "calib_xyz": calib_xyz,
        "p_est_list": p_est_list,
        "p_est_norm_list": p_est_norm_list,
        "p_pdf_display_calc": p_pdf_display_calc,
        "max_errors": max_errors,
    }



def validate_extraction(data, dlt_results):
    print("=== Fonction ExtractionSimu.m : extraction des paramètres (image 1) ===")

    ext = extract_parameters_from_projection(dlt_results["p_est_list"][0], data["spec"].delta_mm)

    print(f"f [mm] calcule         = {ext['f_mm']:.4f}    | PDF: 3.9926")
    print(f"xH [mm] calcule        = {ext['xh_mm']:.4f}    | PDF: 1.8445")
    print(f"yH [mm] calcule        = {ext['yh_mm']:.4f}    | PDF: 2.4597")
    print(f"delta [mm] calcule     = {ext['delta_mm']:.8f} | PDF: {data['spec'].delta_mm:.8f}")
    print(f"x_tilde_O [mm] calcule = {np.array2string(ext['x_tilde_o_mm'], precision=4, suppress_small=False)} | PDF: [-20.017, 0, 330.088] ou meme information signee selon convention")
    print(f"theta_x [deg] calcule  = {ext['theta_x_deg']:.4f} | PDF: 0.0000")
    print(f"theta_y [deg] calcule  = {ext['theta_y_deg']:.4f} | PDF: -45.0150")
    print(f"theta_z [deg] calcule  = {ext['theta_z_deg']:.4f} | PDF: -45.0000")
    print()

    return ext



def validate_measurement_and_reconstruction(data, dlt_results, show_example_m=True):
    print("=== Script Reconstruction3Dsimul.m / ComparaisonSimu.m : mesure 3D ===")

    p_list = dlt_results["p_est_norm_list"]
    recon_points = []
    m13_matrix = None
    x13_recon = None

    for idx in range(81):
        i = idx // 9
        j = idx % 9
        x_rec, m = triangulate_point(
            p_list,
            [data["p"][0][i, j], data["p"][1][i, j], data["p"][2][i, j]],
            [data["q"][0][i, j], data["q"][1][i, j], data["q"][2][i, j]],
        )
        recon_points.append(x_rec)

        if i == 0 and j == 2:
            m13_matrix = m.copy()
            x13_recon = x_rec.copy()

    recon_points = np.array(recon_points)
    theo_points = np.column_stack([data["x_th"].flatten(), data["y_th"].flatten(), data["z_th"].flatten()])

    rho = np.linalg.norm(recon_points - theo_points, axis=1)
    rho_grid = rho.reshape(9, 9)
    mean_um = np.mean(rho) * 1000.0
    max_um = np.max(rho) * 1000.0
    max_idx = int(np.argmax(rho))
    max_i = max_idx // 9 + 1
    max_j = max_idx % 9 + 1

    if show_example_m:
        print_matrix("M pour le point M13", m13_matrix)
        print(f"X(M13) reconstruit [mm] = {np.array2string(x13_recon, precision=4, suppress_small=False)}")
        print(
            "X(M13) theorique [mm]   = "
            + np.array2string(np.array([data['x_th'][0, 2], data['y_th'][0, 2], data['z_th'][0, 2]]), precision=4, suppress_small=False)
        )
        print()

    print(f"Erreur moyenne [um] = {mean_um:.2f} | PDF: 47.7")
    print(f"Erreur max [um]     = {max_um:.2f} | PDF: 106.5")
    print(f"Point erreur max    = M{max_i}{max_j} | PDF: M13")
    print()

    return {
        "recon_points": recon_points,
        "rho_grid": rho_grid,
        "mean_um": mean_um,
        "max_um": max_um,
        "max_label": f"M{max_i}{max_j}",
        "m13_matrix": m13_matrix,
        "x13_recon": x13_recon,
    }



# Entree principale

def run_pdf_validation_suite(show_plots: bool = False):
    data = load_pdf_data()
    direct_results = validate_direct_problem(data)
    dlt_results = validate_dlt_calibration(data, direct_results)
    _ = validate_extraction(data, dlt_results)
    recon_results = validate_measurement_and_reconstruction(data, dlt_results, show_example_m=True)

    if show_plots:
        output_dir = "output_absolute_validation"
        os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(11, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            data["x_th"],
            data["y_th"],
            data["z_th"],
            alpha=0.25,
            cmap="viridis",
            linewidth=0
        )

        calib_xyz, _ = flatten_valid_grid(data["xc"], data["yc"], data["zc"])
        ax1.scatter(
            calib_xyz[:, 0],
            calib_xyz[:, 1],
            calib_xyz[:, 2],
            c="red",
            s=35,
            label="Calibration C"
        )

        ax1.scatter(
            recon_results["recon_points"][:, 0],
            recon_results["recon_points"][:, 1],
            recon_results["recon_points"][:, 2],
            c="magenta",
            s=18,
            label="Reconstructed M"
        )

        ax1.set_title("Shield: theoretical surface + reconstructed points")
        ax1.set_xlabel("X [mm]")
        ax1.set_ylabel("Y [mm]")
        ax1.set_zlabel("Z [mm]")
        ax1.legend(loc="best")

        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(
            recon_results["rho_grid"],
            cmap="YlGnBu_r",
            origin="upper"
        )
        ax2.set_title("Reconstruction error rho [mm]")
        ax2.set_xlabel("j")
        ax2.set_ylabel("i")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()

        fig_path = os.path.join(output_dir, "shield_validation.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    run_pdf_validation_suite(show_plots=True)
