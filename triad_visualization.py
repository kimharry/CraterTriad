import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import combinations
from utils import proj_db2img, sort_clockwise, get_TMC
import numpy as np
from config import SWATH, K

def get_ellipse_params(A, center):
    a = A[0, 0]
    b = 2 * A[0, 1]
    c = A[1, 1]
    d = 2 * A[0, 2]
    e = 2 * A[1, 2]
    f = A[2, 2]

    M = np.array([[a, b/2], [b/2, c]])

    det_M = np.linalg.det(M)
    breakpoint()
    if det_M < 1e-10:
        return None

    M_deriv = 2 * M
    vec_deriv = np.array([-d, -e])

    try:
        x0, y0 = np.linalg.solve(M_deriv, vec_deriv)
    except:
        return None

    eigvals, eigvecs = np.linalg.eig(M)
    lam1, lam2 = eigvals
    
    det_A = np.linalg.det(A)
    if np.abs(det_A) < 1e-10:
        return None
    
    S = -det_A / det_M
    if S/lam1 < 1e-10 or S/lam2 < 1e-10:
        return None
    
    axis1 = np.sqrt(S/lam1)
    axis2 = np.sqrt(S/lam2)
    a, b = max(axis1, axis2), min(axis1, axis2)
    angle = 0.5 * np.arctan2(b, a-c)
    
    return a, b, np.degrees(angle)
    
    


with open('data/filtered_craters_local.pkl', 'rb') as f:
    craters = pickle.load(f)

for comb in combinations(craters, 3):
    c1 = craters[comb[0]]
    c2 = craters[comb[1]]
    c3 = craters[comb[2]]
    
    c1, c2, c3 = sort_clockwise([c1, c2, c3])

    d12 = np.linalg.norm(c1['pos'] - c2['pos'])
    d23 = np.linalg.norm(c2['pos'] - c3['pos'])
    d31 = np.linalg.norm(c3['pos'] - c1['pos'])

    # Swath check
    max_d = SWATH * np.sqrt(2)
    if (d12 > max_d) or (d23 > max_d) or (d31 > max_d):
        continue
    
    # Overlap check
    if (d12 < c1['a'] + c2['a'] + 2) or \
        (d23 < c2['a'] + c3['a'] + 2) or \
        (d31 < c3['a'] + c1['a'] + 2):
        print("Overlapped")

    # seleno_coord = [c1['pos'], c2['pos'], c3['pos']]
    
    center_lat = (c1['lat'] + c2['lat'] + c3['lat']) / 3
    center_lon = (c1['lon'] + c2['lon'] + c3['lon']) / 3
    T_M_C = get_TMC(center_lat, center_lon)

    centers = []
    majors = []
    minors = []
    angles = []
    for crater in [c1, c2, c3]:
        conic = crater['conic_matrix']
        lat, lon = crater['lat'], crater['lon']
        center = R_MOON * np.array([])
        val_at_center = center.T @ conic @ center
        breakpoint()
        if val_at_center >= 0:
            continue
        # conic, center = proj_db2img(T_M_C, crater['pos'], crater['Q_star'])
        # try:
        #     a, b, theta = get_ellipse_params(conic, center[:2])
        # except:
        #     # raise Exception("Ellipse parameters calculation failed")
        #     continue
        # centers.append(tuple(center[:2]))
        # majors.append(a)
        # minors.append(b)
        # angles.append(theta)

    # breakpoint()
    # c1, c2, c3
    fig, ax = plt.subplots(figsize=(10, 10))

    x_lim = [min([centers[0][0] - majors[0]/2, centers[1][0] - majors[1]/2, centers[2][0] - majors[2]/2]) - 1, 
             max([centers[0][0] + majors[0]/2, centers[1][0] + majors[1]/2, centers[2][0] + majors[2]/2]) + 1]
    y_lim = [min([centers[0][1] - majors[0]/2, centers[1][1] - majors[1]/2, centers[2][1] - majors[2]/2]) - 1, 
             max([centers[0][1] + majors[0]/2, centers[1][1] + majors[1]/2, centers[2][1] + majors[2]/2]) + 1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.add_patch(Ellipse(centers[0], majors[0], minors[0], angle=np.degrees(angles[0]), edgecolor='r'))
    ax.add_patch(Ellipse(centers[1], majors[1], minors[1], angle=np.degrees(angles[1]), edgecolor='g'))
    ax.add_patch(Ellipse(centers[2], majors[2], minors[2], angle=np.degrees(angles[2]), edgecolor='b'))

    plt.show()