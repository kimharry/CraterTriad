import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import combinations
from utils import proj_db2img, sort_clockwise, get_TMC, get_center_vector
import numpy as np
from config import SWATH, K, R_MOON, ALTITUDE 

def get_ellipse_params(A_star, center):
    cx, cy = center
    T = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])
    
    # 중심이 (0,0)인 Dual Conic으로 변환
    A_centered = T @ A_star @ T.T
    
    # 3. 스케일 정규화
    # 원점에 있는 표준 Dual Ellipse 식: a^2*u^2 + b^2*v^2 - w^2 = 0
    # 즉, (2,2) 성분이 -1이 되도록 전체를 나눔
    scale = -A_centered[2, 2]
    
    # breakpoint()
    if scale == 0: return None
    
    # 부호가 맞지 않으면(양수면) 허수 타원일 수 있음 -> 절댓값 처리 혹은 None 리턴
    # 여기서는 물리적으로 타원임이 확실하다면 부호를 맞춤
    # if scale < 0: scale = -scale # 강제 부호 보정 (상황에 따라 조정)

    A_norm = A_centered / scale
    
    # 4. 2x2 블록의 고유값 분해
    # 표준형 [a^2, 0; 0, b^2]와 회전행렬의 결합임
    M = A_norm[:2, :2]
    
    eigvals, eigvecs = np.linalg.eig(M)
    
    # 고유값이 음수면 허수 타원 (카메라 뒤쪽 등)
    if np.any(eigvals < 0):
        # 수치 오차로 -0.000... 인 경우 0으로 처리하거나 절대값
        breakpoint()
        eigvals = np.abs(eigvals) 
    
    # Dual Conic의 고유값은 길이의 '제곱' (a^2, b^2)
    axes = np.sqrt(eigvals)
    
    # 정렬 (큰 값이 장축 a, 작은 값이 단축 b)
    sort_idx = np.argsort(axes)[::-1]
    a = axes[sort_idx[0]]
    b = axes[sort_idx[1]]
    
    # 회전각 (장축에 해당하는 고유벡터의 각도)
    # 주의: Dual Space의 고유벡터는 Point Space의 고유벡터와 동일한 방향성을 가짐 (주축)
    v_major = eigvecs[:, sort_idx[0]]
    angle = np.degrees(np.arctan2(v_major[1], v_major[0]))
    
    return a, b, angle


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
        continue

    # seleno_coord = [c1['pos'], c2['pos'], c3['pos']]
    center_lat = (c1['lat'] + c2['lat'] + c3['lat']) / 3
    center_lon = (c1['lon'] + c2['lon'] + c3['lon']) / 3
    T_M_C = get_TMC(center_lat, center_lon)
    r_M = get_center_vector(center_lat, center_lon, r=R_MOON + ALTITUDE)

    centers = []
    majors = []
    minors = []
    angles = []
    for crater in [c1, c2, c3]:
        conic, center = proj_db2img(T_M_C, r_M, crater['Q_star'])
        a, b, theta = get_ellipse_params(conic, center[:2])
        centers.append(tuple(center[:2]))
        majors.append(a)
        minors.append(b)
        angles.append(theta)

    breakpoint()

    # c1, c2, c3
    # fig, ax = plt.subplots(figsize=(10, 10))

    # x_lim = [min([centers[0][0] - majors[0]/2, centers[1][0] - majors[1]/2, centers[2][0] - majors[2]/2]) - 1, 
    #          max([centers[0][0] + majors[0]/2, centers[1][0] + majors[1]/2, centers[2][0] + majors[2]/2]) + 1]
    # y_lim = [min([centers[0][1] - majors[0]/2, centers[1][1] - majors[1]/2, centers[2][1] - majors[2]/2]) - 1, 
    #          max([centers[0][1] + majors[0]/2, centers[1][1] + majors[1]/2, centers[2][1] + majors[2]/2]) + 1]
    # ax.set_xlim(x_lim)
    # ax.set_ylim(y_lim)
    # ax.add_patch(Ellipse(centers[0], majors[0], minors[0], angle=angles[0], edgecolor='r'))
    # ax.add_patch(Ellipse(centers[1], majors[1], minors[1], angle=angles[1], edgecolor='g'))
    # ax.add_patch(Ellipse(centers[2], majors[2], minors[2], angle=angles[2], edgecolor='b'))

    # plt.show()