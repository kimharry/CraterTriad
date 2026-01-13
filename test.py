# 수동으로 간단한 케이스 테스트
import numpy as np

# 원형 크레이터 3개 (단순화)
def test_simple_case():
    # Identity matrix로 간단한 원 3개
    A1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, -1]])  # x^2 + y^2 - 1 = 0
    
    A2 = np.array([[1, 0, -2],
                   [0, 1, 0],
                   [-2, 0, 3]])  # (x-2)^2 + y^2 - 1 = 0
    
    A3 = np.array([[1, 0, 0],
                   [0, 1, -2],
                   [0, -2, 3]])  # x^2 + (y-2)^2 - 1 = 0
    
    # Envelope (inverse)
    A1_star = np.linalg.inv(A1)
    A2_star = np.linalg.inv(A2)
    A3_star = np.linalg.inv(A3)
    
    # Test line: x = 1 (수직선)
    l = np.array([[1], [0], [-1]])
    
    print("Simple test:")
    print(f"  l^T A1* l = {(l.T @ A1_star @ l).item():.6f}")
    print(f"  l^T A2* l = {(l.T @ A2_star @ l).item():.6f}")
    
    # α 계산
    l1 = np.array([[1], [0], [0]])  # x = 0
    l2 = np.array([[0], [1], [0]])  # y = 0
    
    num = (l1.T @ A1_star @ l2).item()
    den1 = (l1.T @ A1_star @ l1).item()
    den2 = (l2.T @ A1_star @ l2).item()
    
    alpha = abs(num) / np.sqrt(den1 * den2)
    print(f"\n  numerator = {num:.6f}")
    print(f"  den1 = {den1:.6f}, den2 = {den2:.6f}")
    print(f"  α = {alpha:.6f}")
    print(f"  Should be ≥ 1? {alpha >= 1}")

test_simple_case()