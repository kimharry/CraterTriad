import csv
import numpy as np
from pose_estimation import main as pose_main
from config import T_M_C


if __name__ == "__main__":
    detections = []
    
    with open('../CDA/output/test/ellipse/ellipse_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections.append({
                'pos': np.array([float(row['cx_px']), float(row['cy_px'])]),
                'a': float(row['major_axis_px']),
                'b': float(row['minor_axis_px']),
                'theta': np.radians(float(row['angle_deg']))
            })

    # Run pose estimation
    r_M = pose_main(detections, T_M_C)
    print("Estimated pose:", r_M)

