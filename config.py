import numpy as np

R_MOON = 1737.4

# location info
MIN_LAT = 43.4
MAX_LAT = 46.7
MIN_LON = 191.4
MAX_LON = 196.2

# major axis size limit
MAX_AXIS_KM = 100.0

# quality limit
# ARC_IMG: rim arc ratio
# PTS_RIM_IMG: number of points used for fitting
MIN_ARC = 0.8
MIN_PTS = 5

LAT_AVG = (MIN_LAT + MAX_LAT) / 2
LON_AVG = (MIN_LON + MAX_LON) / 2

# SWATH = 1.88 # Altitude: 3.5km
# SWATH = 8.04 # Altitude: 15km
ALTITUDE = 50 # km
FOV = 30 # degrees
SWATH = ALTITUDE * np.tan(np.radians(FOV/2)) * 2

K = np.array([
    [1866.03, 0.0, 500.0],
    [0.0, 1866.03, 500.0],
    [0.0, 0.0, 1.0]
])

# T_M_C = np.array([
#         [0.0, 0.0, -1.0],
#         [0.0, 1.0, 0.0],
#         [1.0, 0.0, 0.0]
#     ])

# Nadir-pointing, ENU-aligned
T_M_C = np.array([
    [ 0.23853346,  0.68729463,  0.68609612],
    [-0.97113428,  0.16881575,  0.16852137],
    [ 0.        ,  0.70648944, -0.70772358]
])