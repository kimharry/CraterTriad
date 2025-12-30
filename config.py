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

LAT_AVG = 45.05
LON_AVG = 193.8

# SWATH = 1.88 # Altitude: 3.5km
# SWATH = 8.04 # Altitude: 15km
ALTITUDE = 50 # km
FOV = 30 # degrees
SWATH = ALTITUDE * np.tan(np.radians(FOV/2)) * 2