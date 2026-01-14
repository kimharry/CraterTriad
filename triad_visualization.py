import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from itertools import combinations
from utils import proj_db2img, sort_clockwise, get_center_vector, get_ellipse_params, get_TMC
import numpy as np
from config import SWATH, R_MOON, ALTITUDE

def plot_3d_quadric(ax, p_M, T_E_M, a, b, theta, color='b'):
    """
    p_M: Moon-centered position (3,)
    T_E_M: ENU to Moon rotation matrix (3x3) - [East, North, Up] vectors
    a, b: Semi-major, Semi-minor axis lengths
    theta: Rotation angle of major axis from East (radians)
    """
    
    # 1. 2D 타원 좌표 생성 (로컬 평면)
    t = np.linspace(0, 2 * np.pi, 100)
    
    # 파라메트릭 방정식 (회전 포함)
    # x_local = a * cos(t) * cos(theta) - b * sin(t) * sin(theta)
    # y_local = a * cos(t) * sin(theta) + b * sin(t) * cos(theta)
    
    # 행렬 형태로 깔끔하게 계산
    # 로컬 원 -> 스케일링 -> 회전
    circle_points = np.stack([np.cos(t), np.sin(t)], axis=0)
    scale_matrix = np.diag([a, b])
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Local ENU 좌표계에서의 점들 (2x100)
    local_points_2d = rot_matrix @ scale_matrix @ circle_points
    
    # 3D 로컬 좌표로 확장 (z=0, 즉 Up 방향 성분은 0)
    # (3x100) 형태: [East, North, Up]
    local_points_3d = np.vstack([
        local_points_2d, 
        np.zeros(100) 
    ])
    
    # 2. 3D 월드 좌표로 변환 (Rigid Body Transform)
    # World_Point = R_local_to_world @ Local_Point + Center_Position
    # T_E_M이 바로 로컬(ENU)을 월드(Moon)로 보내는 회전 행렬입니다.
    world_points = T_E_M @ local_points_3d + p_M.reshape(3, 1)
    
    # 3. 플로팅
    ax.plot(world_points[0, :], world_points[1, :], world_points[2, :], color=color, linewidth=2)
    
    # (선택) 중심점 및 법선 벡터(Up vector) 시각화
    ax.scatter(*p_M, color='r', s=20) # 중심
    
    # 법선 벡터 (Up direction check)
    normal_start = p_M
    normal_end = p_M + T_E_M[:, 2] * (a + b) / 2  # 적당한 길이로
    ax.plot([normal_start[0], normal_end[0]],
            [normal_start[1], normal_end[1]],
            [normal_start[2], normal_end[2]], color='g', linestyle='--')


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
    if (d12+c1['a']+c2['a'] > SWATH) or (d23+c2['a']+c3['a'] > SWATH) or (d31+c3['a']+c1['a'] > SWATH):
        continue
    
    # Overlap check
    if (d12 < c1['a'] + c2['a']) or \
        (d23 < c2['a'] + c3['a']) or \
        (d31 < c3['a'] + c1['a']):
        continue

    # breakpoint()
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
        A_star, A = proj_db2img(T_M_C, r_M, crater['Q_star'])
        params = get_ellipse_params(A)
        if params is None:
            continue
        centers.append(params['center'])
        majors.append(params['major'])
        minors.append(params['minor'])
        angles.append(params['angle'])

    if len(centers) != 3:
        continue

    # c1, c2, c3
    fig, ax = plt.subplots(figsize=(10, 10))
    # x_lim = np.array([min(center[0] - major for center, major in zip(centers, majors)), \
    #                   max(center[0] + major for center, major in zip(centers, majors))])
    # y_lim = np.array([max(center[1] + minor for center, minor in zip(centers, minors)), \
    #                   min(center[1] - minor for center, minor in zip(centers, minors))])
    ax.set_xlim([0, 1000])
    ax.set_ylim([1000, 0])
    ax.add_patch(Ellipse(centers[0], 2*majors[0], 2*minors[0], angle=angles[0], fill=False, edgecolor='r'))
    ax.add_patch(Ellipse(centers[1], 2*majors[1], 2*minors[1], angle=angles[1], fill=False, edgecolor='g'))
    ax.add_patch(Ellipse(centers[2], 2*majors[2], 2*minors[2], angle=angles[2], fill=False, edgecolor='b'))
    ax.grid()
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_quadric(ax, c1['pos'], c1['T_E_M'], c1['a'], c1['b'], c1['theta'], color='r')
    plot_3d_quadric(ax, c2['pos'], c2['T_E_M'], c2['a'], c2['b'], c2['theta'], color='g')
    plot_3d_quadric(ax, c3['pos'], c3['T_E_M'], c3['a'], c3['b'], c3['theta'], color='b')
    plt.show()