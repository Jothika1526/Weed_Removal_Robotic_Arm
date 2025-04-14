import numpy as np

def dh_transform_modified(a, alpha, d, theta):
    """Compute the transformation matrix using Modified DH parameters."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])

def fk_solver(joint_angles):
    """Compute forward kinematics using joint angles."""
    if len(joint_angles) < 4:
        raise ValueError("Not enough joint angles received!")

    theta1 = joint_angles[0]
    theta2 = joint_angles[1]
    theta3 = joint_angles[2]
    theta4 = joint_angles[3]

    # Fixed transformation from base to link1 frame (T0)
    T0 = np.array([
        [1, 0, 0, 0.012],  # x-offset
        [0, 1, 0, 0.0],    # y-offset
        [0, 0, 1, 0.017],  # z-offset
        [0, 0, 0, 1]
    ])

    # Transformation due to Joint 1 (θ1) — about Y-axis
    T1 = dh_transform_modified(a=0.0, alpha=0.0, d=0.0, theta=theta1)

    # Fixed transform from link2 to joint2 frame (translation only)
    T_link2_to_joint2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.0595],
        [0, 0, 0, 1]
    ])

    # Rotation about Y-axis for Joint2 (θ2)
    T_joint2 = np.array([
        [np.cos(theta2), 0, np.sin(theta2), 0],
        [0, 1, 0, 0],
        [-np.sin(theta2), 0, np.cos(theta2), 0],
        [0, 0, 0, 1]
    ])

    # Total FK up to Link3
    T_to_link3 = T0 @ T1 @ T_link2_to_joint2 @ T_joint2

    # ------- FK Extension: Link3 → Link4 via Joint3 -------
    # Fixed transform from link3 to joint3 frame (translation only)
    T_link3_to_joint3 = np.array([
        [1, 0, 0, 0.024],
        [0, 1, 0, 0],
        [0, 0, 1, 0.128],
        [0, 0, 0, 1]
    ])

    # Rotation about Y-axis for Joint3 (θ3)
    T_joint3 = np.array([
        [np.cos(theta3), 0, np.sin(theta3), 0],
        [0, 1, 0, 0],
        [-np.sin(theta3), 0, np.cos(theta3), 0],
        [0, 0, 0, 1]
    ])

    # Total FK up to Link4
    T_to_link4 = T_to_link3 @ T_link3_to_joint3 @ T_joint3

    # ------- FK Extension: Link4 → Link5 via Joint4 -------
    # Fixed transform from link4 to joint4 frame (translation only)
    T_link4_to_joint4 = np.array([
        [1, 0, 0, 0.124],  # x-offset (same as in your URDF)
        [0, 1, 0, 0.0],    # y-offset
        [0, 0, 1, 0.0],    # z-offset
        [0, 0, 0, 1]
    ])

    # Rotation about Y-axis for Joint4 (θ4)
    T_joint4 = np.array([
        [np.cos(theta4), 0, np.sin(theta4), 0],
        [0, 1, 0, 0],
        [-np.sin(theta4), 0, np.cos(theta4), 0],
        [0, 0, 0, 1]
    ])

    # Total FK up to Link5
    T_to_link5 = T_to_link4 @ T_link4_to_joint4 @ T_joint4

    # ------- FK Extension: Link5 → End Effector -------
    # Fixed transformation from link5 to end effector (translation only, from URDF)
    T_link5_to_end_effector = np.array([
        [1, 0, 0, 0.126],  # x-offset (from URDF)
        [0, 1, 0, 0.0],    # No offset along y-axis
        [0, 0, 1, 0.0],    # No offset along z-axis
        [0, 0, 0, 1]
    ])

    # Total FK from base to end effector
    T_to_end_effector = T_to_link5 @ T_link5_to_end_effector

    # ---------------- Return Outputs ----------------
    return T_to_end_effector


