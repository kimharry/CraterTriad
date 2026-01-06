from utils import get_ENU_to_Moon_matrix
import numpy as np
from config import R_MOON, LAT_AVG, LON_AVG, ALTITUDE

def get_center_vector(lat, lon):
    return (R_MOON + ALTITUDE) * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

p_M = get_center_vector(np.radians(LAT_AVG), np.radians(LON_AVG))
print(p_M)

T_M_C = get_ENU_to_Moon_matrix(p_M)
T_M_C[:, 2] = T_M_C[:, 2] * -1
print(T_M_C)