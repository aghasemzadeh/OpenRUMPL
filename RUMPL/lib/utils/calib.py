# ------------------------------------------------------------------------------
# Copyright (c) 2024 UCLouvain. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
#
# Author: Seyed Abolfazl Ghaemzadeh, ICTEAM, UCLouvain
# ------------------------------------------------------------------------------
'''
This file contains functions for camera calibration and skew lines distance calculation.
'''

import numpy as np
import copy

def find_3_points_on_ray(u,v,fx,fy,ox,oy):
    '''
    u, v: pixel coordinates shape (b,)
    '''
    assert u.shape == v.shape
    
    x = np.stack([(u-ox)/fx, np.ones(u.shape), (fy/fx)*((u-ox)/(v-oy))])
    y = np.stack([(v-oy)/fy, (fx/fy)*((v-oy)/(u-ox)), np.ones(u.shape)])
    z = np.stack([np.ones(u.shape), fx/(u-ox), fy/(v-oy)])
    P = np.stack([x, y, z])
    P = np.transpose(P)
    return P

def cam_to_world(point_cam, R, T):
    # takes batches of points and converts them to world coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, 3, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E_inv = np.linalg.inv(E)
    E_inv = np.stack([E_inv]*point_cam.shape[1])                      # create E_inv for 3 points
    E_inv = np.stack([E_inv]*p.shape[0])             # create E_inv for all matchings
    point_world = E_inv @ p
    point_world_squeezed = np.squeeze(point_world, axis=-1)
    point_world = point_world_squeezed[:,:,:] / point_world[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_world = point_world[:,:,:3]
    return point_world

def world_to_cam(point_world, R, T):
    # takes batches of points and converts them to world coordinates
    # point_world: (b, n, 3) batch, number of points, xyz
    # build E
    R = np.array(R)
    to_conc = np.ones((point_world.shape[0], point_world.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_world, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    R_temp = np.vstack([R, np.zeros((1,3))])
    T_temp = np.vstack([T, 1.0])
    E = np.hstack([R_temp, T_temp])
    E = np.stack([E]*point_world.shape[1])                      # create E for 17 points
    E = np.stack([E]*p.shape[0])             # create E for all matchings
    point_cam = E @ p
    point_cam_squeezed = np.squeeze(point_cam, axis=-1)
    point_cam = point_cam_squeezed[:,:,:] / point_cam[:,:,3]  # convert back to cartesian (check that x_homo[2] > 0)
    point_cam = point_cam[:,:,:3]
    return point_cam

def cam_to_image(point_cam, K):
    # takes batches of points and converts them to image 2D coordinates
    # point_cam: (b, n, 3) batch, number of points, xyz
    
    K = np.array(K)
    to_conc = np.ones((point_cam.shape[0], point_cam.shape[1], 1))     # this for concatenating 
    p = np.concatenate([point_cam, to_conc], axis=2)  # convert to homogenous         (b, n, 4) batch, number of points, homogenous coordinates
    p = np.expand_dims(p, axis=3)
    M = np.hstack([K, np.zeros((3, 1))])
    M = np.stack([M]*point_cam.shape[1])                      # create M for 17 points
    M = np.stack([M]*p.shape[0])             # create M for all matchings
    
    point_img = M @ p
    point_img_squeezed = np.squeeze(point_img, axis=-1)
    point_img = point_img_squeezed[:,:,:] / point_img[:,:,2]  # convert back to cartesian (check that x_homo[1] > 0)
    point_img = point_img[:,:,:2]
    return point_img



def find_line_with_3points(points):
    K = np.array(
        [[points[0][1], points[0][2], -1],
        [points[1][1], points[1][2], -1],
        [points[2][1], points[2][2], -1]])
    print(K.shape)
    Y = np.array(
        [-points[0][0], -points[1][0], -points[2][0]])
    print(Y.shape)
    K_inv = np.linalg.inv(K)
    X = K_inv @ Y
    return X
    
def distance_between_two_skew_lines(points_l1, points_l2):
    
    nominator = np.array(
    [[points_l2[:, 0, 0] - points_l1[:, 0, 0], points_l2[:, 0, 1] - points_l1[:, 0, 1], points_l2[:, 0, 2] - points_l1[:, 0, 2]],
     [points_l1[:, 1, 0] - points_l1[:, 2, 0], points_l1[:, 1, 1] - points_l1[:, 2, 1], points_l1[:, 1, 2] - points_l1[:, 2, 2]],
     [points_l2[:, 1, 0] - points_l2[:, 2, 0], points_l2[:, 1, 1] - points_l2[:, 2, 1], points_l2[:, 1, 2] - points_l2[:, 2, 2]]]
    )

    nominator = np.transpose(nominator,(2,0,1))

    det = np.linalg.det(nominator)
    A = (points_l1[:, 1, 1] - points_l1[:, 2, 1]) * (points_l2[:, 1, 2] - points_l2[:, 2, 2]) - (points_l2[:, 1, 1] - points_l2[:, 2, 1]) * (points_l1[:, 1, 2] - points_l1[:, 2, 2])
    B = (points_l1[:, 1, 2] - points_l1[:, 2, 2]) * (points_l2[:, 1, 0] - points_l2[:, 2, 0]) - (points_l2[:, 1, 2] - points_l2[:, 2, 2]) * (points_l1[:, 1, 0] - points_l1[:, 2, 0])
    C = (points_l1[:, 1, 0] - points_l1[:, 2, 0]) * (points_l2[:, 1, 1] - points_l2[:, 2, 1]) - (points_l2[:, 1, 0] - points_l2[:, 2, 0]) * (points_l1[:, 1, 1] - points_l1[:, 2, 1])    

    
    denominator = np.sqrt(A**2 + B**2 + C**2)

    
    return np.abs(det/denominator)


def epiolar_error_between_cams(camera_0, camera_1, points_0, points_1, points_0_conf, points_1_conf):
    # camera_0, camera_1: dict with K, R and t
    # points_0, points_1: (n, 2) 2D points in image coordinates

    cam_0_fx = camera_0['fx']
    cam_0_fy = camera_0['fy']
    cam_0_ox = camera_0['cx']
    cam_0_oy = camera_0['cy']
    cam_0_R = camera_0['R']
    cam_0_t = camera_0['t']

    u, v = (points_0[:, 0].reshape(-1,), points_0[:, 1].reshape(-1,))
    P0 = find_3_points_on_ray(u, v, cam_0_fx, cam_0_fy, cam_0_ox, cam_0_oy)  # output shape (b,3,3)

    cam_1_fx = camera_1['fx']
    cam_1_fy = camera_1['fy']
    cam_1_ox = camera_1['cx']
    cam_1_oy = camera_1['cy']
    cam_1_R = camera_1['R']
    cam_1_t = camera_1['t']
    u, v = (points_1[:, 0].reshape(-1,), points_1[:, 1].reshape(-1,))
    P1 = find_3_points_on_ray(u, v, cam_1_fx, cam_1_fy, cam_1_ox, cam_1_oy)

    P0 = cam_to_world(P0, cam_0_R, cam_0_t)
    P1 = cam_to_world(P1, cam_1_R, cam_1_t)

    epipolar_error = distance_between_two_skew_lines(P0, P1)

    min_conf = np.minimum(points_0_conf, points_1_conf)

    epipolar_error = epipolar_error * min_conf.reshape(-1, 1)

    epipolar_error = np.nansum(epipolar_error, axis=1) / np.nansum(min_conf, axis=1)

    
    epipolar_error = np.nanmean(epipolar_error, axis=0)

    if np.isnan(epipolar_error):
        # If there are NaN values, set them to 1000000
        epipolar_error = 1000000

    return epipolar_error

def smart_pseudo_remove_weight(target, weight, meta, epipolar_error_threshold=5):
    '''
    The purpose is to calculate the epipolar error between each pair of cameras.
    And then, if the epipolar error is larger than 5, we set the weight of the corresponding keypoint to 0.
    
    target: list, element: (17, 3)
    weight: list, element: (17,)
    meta: list, element: dict
    '''
    n_views = len(target)
    n_kps = meta[0]['joints_2d'].shape[0]
    target = copy.deepcopy(target)
    weight = copy.deepcopy(weight)
    meta = copy.deepcopy(meta)
    
    epipolar_error_by_cams = np.zeros((n_views, n_kps))
    for i in range(n_views):
        for j in range(n_views):
            if i >= j:
                continue
            else:
                cam_0_fx = meta[i]['camera']['fx']
                cam_0_fy = meta[i]['camera']['fy']
                cam_0_ox = meta[i]['camera']['cx']
                cam_0_oy = meta[i]['camera']['cy']
                u, v = (meta[i]['joints_2d'][:,0].reshape(-1,), meta[i]['joints_2d'][:,1].reshape(-1,))
                P0 = find_3_points_on_ray(u,v,cam_0_fx,cam_0_fy,cam_0_ox,cam_0_oy)               # output shape (b,3,3)
    
                cam_1_fx = meta[j]['camera']['fx']
                cam_1_fy = meta[j]['camera']['fy']
                cam_1_ox = meta[j]['camera']['cx']
                cam_1_oy = meta[j]['camera']['cy']
                u, v = (meta[j]['joints_2d'][:,0].reshape(-1,), meta[j]['joints_2d'][:,1].reshape(-1,))
    
                P1 = find_3_points_on_ray(u,v,cam_1_fx,cam_1_fy,cam_1_ox,cam_1_oy)
    
                cam_0_R = meta[i]['camera']['R']
                cam_0_t = meta[i]['camera']['t']
                cam_1_R = meta[j]['camera']['R']
                cam_1_t = meta[j]['camera']['t']
    
                P0 = cam_to_world(P0, cam_0_R, cam_0_t)
                P1 = cam_to_world(P1, cam_1_R, cam_1_t)
    
                epipolar_error = distance_between_two_skew_lines(P0,P1)
                
                epipolar_error_by_cams[i] += epipolar_error * meta[i]['joints_2d_conf'].reshape(-1,)
                epipolar_error_by_cams[j] += epipolar_error * meta[j]['joints_2d_conf'].reshape(-1,)
    
    epipolar_error_by_cams = epipolar_error_by_cams / (n_views-1)
    
    for i in range(n_views):
        weight[i][epipolar_error_by_cams[i] > epipolar_error_threshold] = 0
    return weight
    

def epiolar_error_between_cams(camera_0, camera_1, points_0, points_1, points_0_conf, points_1_conf):
    # camera_0, camera_1: dict with K, R and t
    # points_0, points_1: (n, 2) 2D points in image coordinates

    cam_0_fx = camera_0['fx']
    cam_0_fy = camera_0['fy']
    cam_0_ox = camera_0['cx']
    cam_0_oy = camera_0['cy']
    cam_0_R = camera_0['R']
    cam_0_t = camera_0['t']

    u, v = (points_0[:, 0].reshape(-1,), points_0[:, 1].reshape(-1,))
    P0 = find_3_points_on_ray(u, v, cam_0_fx, cam_0_fy, cam_0_ox, cam_0_oy)  # output shape (b,3,3)

    cam_1_fx = camera_1['fx']
    cam_1_fy = camera_1['fy']
    cam_1_ox = camera_1['cx']
    cam_1_oy = camera_1['cy']
    cam_1_R = camera_1['R']
    cam_1_t = camera_1['t']
    u, v = (points_1[:, 0].reshape(-1,), points_1[:, 1].reshape(-1,))
    P1 = find_3_points_on_ray(u, v, cam_1_fx, cam_1_fy, cam_1_ox, cam_1_oy)

    P0 = cam_to_world(P0, cam_0_R, cam_0_t)
    P1 = cam_to_world(P1, cam_1_R, cam_1_t)

    epipolar_error = distance_between_two_skew_lines(P0, P1)

    min_conf = np.minimum(points_0_conf, points_1_conf)

    epipolar_error = epipolar_error * min_conf.reshape(-1, 1)

    epipolar_error = np.nansum(epipolar_error, axis=1) / np.nansum(min_conf, axis=1)

    
    epipolar_error = np.nanmean(epipolar_error, axis=0)

    if np.isnan(epipolar_error):
        # If there are NaN values, set them to 1000000
        epipolar_error = 1000000

    return epipolar_error





def smart_pseudo_remove_weight(target, weight, meta, epipolar_error_threshold=5):
    '''
    The purpose is to calculate the epipolar error between each pair of cameras.
    And then, if the epipolar error is larger than 5, we set the weight of the corresponding keypoint to 0.
    
    target: list, element: (17, 3)
    weight: list, element: (17,)
    meta: list, element: dict
    '''
    n_views = len(target)
    n_kps = meta[0]['joints_2d'].shape[0]
    target = copy.deepcopy(target)
    weight = copy.deepcopy(weight)
    meta = copy.deepcopy(meta)
    
    epipolar_error_by_cams = np.zeros((n_views, n_kps))
    for i in range(n_views):
        for j in range(n_views):
            if i >= j:
                continue
            else:
                cam_0_fx = meta[i]['camera']['fx']
                cam_0_fy = meta[i]['camera']['fy']
                cam_0_ox = meta[i]['camera']['cx']
                cam_0_oy = meta[i]['camera']['cy']
                u, v = (meta[i]['joints_2d'][:,0].reshape(-1,), meta[i]['joints_2d'][:,1].reshape(-1,))
                P0 = find_3_points_on_ray(u,v,cam_0_fx,cam_0_fy,cam_0_ox,cam_0_oy)               # output shape (b,3,3)
    
                cam_1_fx = meta[j]['camera']['fx']
                cam_1_fy = meta[j]['camera']['fy']
                cam_1_ox = meta[j]['camera']['cx']
                cam_1_oy = meta[j]['camera']['cy']
                u, v = (meta[j]['joints_2d'][:,0].reshape(-1,), meta[j]['joints_2d'][:,1].reshape(-1,))
    
                P1 = find_3_points_on_ray(u,v,cam_1_fx,cam_1_fy,cam_1_ox,cam_1_oy)
    
                cam_0_R = meta[i]['camera']['R']
                cam_0_t = meta[i]['camera']['t']
                cam_1_R = meta[j]['camera']['R']
                cam_1_t = meta[j]['camera']['t']
    
                P0 = cam_to_world(P0, cam_0_R, cam_0_t)
                P1 = cam_to_world(P1, cam_1_R, cam_1_t)
    
                epipolar_error = distance_between_two_skew_lines(P0,P1)
                
                epipolar_error_by_cams[i] += epipolar_error * meta[i]['joints_2d_conf'].reshape(-1,)
                epipolar_error_by_cams[j] += epipolar_error * meta[j]['joints_2d_conf'].reshape(-1,)
    
    epipolar_error_by_cams = epipolar_error_by_cams / (n_views-1)
    
    for i in range(n_views):
        weight[i][epipolar_error_by_cams[i] > epipolar_error_threshold] = 0
    return weight
    
def closest_point_between_rays_batched(batch_points, batch_directions, batch_confidences):
    """
    Find the closest point of intersection for a batch of multiple sets of rays using vectorized operations.

    Args:
    - batch_points: A NumPy array of shape (B, N, 3) where B is the batch size and N is the number of rays per batch.
    - batch_directions: A NumPy array of shape (B, N, 3) containing the direction vectors for the rays in each batch.
    - batch_confidences: A NumPy array of shape (B, N) containing the confidence values for each ray in each batch.

    Returns:
    - A NumPy array of shape (B, 3) containing the intersection points for each batch.
    """

    # Normalize the confidence values for each batch
    batch_confidences = batch_confidences / np.sum(batch_confidences, axis=1, keepdims=True)

    # Normalize direction vectors to unit vectors
    batch_directions = batch_directions / np.linalg.norm(batch_directions, axis=2, keepdims=True)

    # Create the identity matrix of shape (3, 3) and expand it to be (B, N, 3, 3)
    I = np.eye(3)
    I = np.expand_dims(I, axis=0)  # Shape (1, 3, 3)
    
    # Calculate the projection matrices (B, N, 3, 3)
    # Projection matrix for each direction: I - dir * dir^T
    outer_products = np.einsum('bij,bik->bijk', batch_directions, batch_directions)  # Outer product of directions
    projection_matrices = I - outer_products  # Shape (B, N, 3, 3)

    # Apply the confidences: reshape to (B, N, 1, 1) for broadcasting
    weighted_projection_matrices = projection_matrices * batch_confidences[:, :, np.newaxis, np.newaxis]  # Shape (B, N, 3, 3)

    # Sum over the rays (axis=1), resulting in (B, 3, 3)
    A_matrices = np.sum(weighted_projection_matrices, axis=1)

    # Calculate the weighted b vector for each ray
    # b = confidence * projection_matrix * point
    weighted_b = np.einsum('bijk,bik->bij', projection_matrices, batch_points)  # Shape (B, N, 3)
    weighted_b = weighted_b * batch_confidences[:, :, np.newaxis]  # Shape (B, N, 3)

    # Sum over the rays (axis=1), resulting in (B, 3)
    b_vectors = np.sum(weighted_b, axis=1)

    # Solve the system of linear equations A * x = b for each batch element (B, 3)
    intersection_points = np.linalg.solve(A_matrices, b_vectors)

    return intersection_points

def skew_symmetric_matrix(t):
    """
    Returns the skew-symmetric matrix of a vector t.
    """
    t = np.array(t)
    t = np.squeeze(t)
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def compute_relative_extrinsics(R1, t1, R2, t2):
    """
    Computes the relative rotation and translation between two cameras given
    their extrinsic parameters.
    
    Args:
        R1: Rotation matrix of the first camera.
        t1: Translation vector of the first camera.
        R2: Rotation matrix of the second camera.
        t2: Translation vector of the second camera.
    
    Returns:
        R_rel: Relative rotation matrix between the two cameras.
        t_rel: Relative translation vector between the two cameras.
    """
    R1 = np.array(R1)
    t1 = np.array(t1)
    R2 = np.array(R2)
    t2 = np.array(t2)
    # Relative rotation
    R_rel = R2 @ R1.T
    # Relative translation
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel

def compute_fundamental_matrix(K1, K2, R_rel, t_rel):
    """
    Computes the fundamental matrix given intrinsic matrices K1, K2, relative rotation R_rel, and relative translation t_rel.
    """
    t_cross = skew_symmetric_matrix(t_rel)
    F = np.linalg.inv(K2).T @ t_cross @ R_rel @ np.linalg.inv(K1)
    return F

def find_epipolar_lines(F, points):
    """
    Computes the epipolar lines in the second view for a batch of points in the first view.
    
    Args:
        F: The 3x3 fundamental matrix.
        points: An Nx3 matrix, where each row is a point in the first view (in homogeneous coordinates).
    
    Returns:
        lines: An Nx3 matrix, where each row is the epipolar line parameters in the second view.
    """
    # Ensure points are in the correct shape (Nx3)
    if points.ndim == 1:
        points = points[np.newaxis, :]  # Convert to 2D if a single point is provided
    
    # Compute the epipolar lines in the second view for each point
    lines = points @ F.T
    lines = lines.T
    return lines

def compute_intersection(l1, l2):
    """
    Computes the intersection of two lines in homogeneous coordinates using the cross product.
    
    Args:
        l1: The first line in homogeneous coordinates (3x1 array).
        l2: The second line in homogeneous coordinates (3x1 array).
    
    Returns:
        p: The intersection point in homogeneous coordinates (3x1 array).
    """
    # Compute the cross product of the two lines to find the intersection point
    p = np.cross(l1, l2)
    
    # Normalize to ensure homogeneous coordinates (if not at infinity)
    if p[2] != 0:
        p = p / p[2]
    
    return p

def find_intersections(lines):
    """
    Finds the intersections between each pair of lines for a set of lines.
    
    Args:
        lines: A list of lines (Nx3 array, where N is the number of lines).
    
    Returns:
        intersections: A list of intersection points in homogeneous coordinates.
    """
    intersections = []
    num_lines = len(lines)
    
    # Compute intersections for each pair of lines
    for i in range(num_lines):
        for j in range(i+1, num_lines):
            intersection = compute_intersection(lines[i], lines[j])
            intersections.append(intersection)
    
    return intersections

def find_intersections_for_keypoints(keypoints_lines):
    """
    Computes the intersections for each keypoint, where each keypoint has a set of lines.
    
    Args:
        keypoints_lines: A list of lists, where each sublist contains the lines for a keypoint.
    
    Returns:
        keypoints_intersections: A list of lists, where each sublist contains the intersections
                                 for the corresponding keypoint.
    """
    keypoints_intersections = []
    
    for lines in keypoints_lines:
        intersections = find_intersections(lines)
        keypoints_intersections.append(intersections)
    
    return keypoints_intersections

def closest_points_for_keypoints(keypoints_lines):
    """
    Finds the closest point to each set of lines for 17 keypoints.

    Parameters:
        keypoints_lines (list): A list of 17 entries. Each entry is a list of tuples where
                                each tuple contains two numpy arrays:
                                - a point on the line (numpy array of shape (3,))
                                - the direction vector of the line (numpy array of shape (3,))
    
    Returns:
        list: A list of 17 numpy arrays, each representing the closest point to the corresponding set of lines.
    """
    def closest_point_least_squares(lines):
        """Helper function to compute closest point for a set of lines."""
        A = np.zeros((3, 3))
        b = np.zeros(3)

        for P_i, D_i in lines:
            D_i = D_i / np.linalg.norm(D_i)  # Normalize direction vector
            A += np.eye(3) - np.outer(D_i, D_i)
            b += (np.eye(3) - np.outer(D_i, D_i)).dot(P_i)

        # Solve the linear system A * x = b for x
        closest_point = np.linalg.solve(A, b)
        return closest_point

    # Calculate the closest point for each keypoint's set of lines
    closest_points = []
    for keypoint_lines in keypoints_lines:
        closest_points.append(closest_point_least_squares(keypoint_lines))
    
    return closest_points


# Function to find the intersection of two lines given in ax + by + c = 0 form
def intersection(line1, line2):
    # Extract coefficients a, b, and c for both lines
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    # Create coefficient matrix and constant vector
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([-c1, -c2])
    
    # Solve the system of linear equations
    try:
        intersect = np.linalg.solve(A, B)
        return intersect
    except np.linalg.LinAlgError:
        # If lines are parallel or coincident, there is no single intersection
        return None

# Function to find the centroid of points
def centroid(points):
    points = np.array(points)
    if len(points) == 0:
        return None
    return np.mean(points, axis=0)

# Main function to compute the middle point of each set of lines
def middle_point_of_sets(line_sets):
    middle_points = []
    
    for lines in line_sets:
        intersection_points = []
        
        # Find all intersections between pairs of lines
        num_lines = len(lines)
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                point = intersection(lines[i], lines[j])
                if point is not None:
                    intersection_points.append(point)
        
        # Calculate the centroid of intersection points
        middle_point = centroid(intersection_points)
        middle_points.append(middle_point)
        
    middle_points = np.array(middle_points)
    
    return middle_points

import numpy as np

def closest_points_on_n_skew_lines(points, directions):
    """
    Find the closest points on n skew lines in 3D.
    
    Parameters:
    - points: List of points (one on each line), shape (n, 3)
    - directions: List of direction vectors (one for each line), shape (n, 3)
    
    Returns:
    - closest_points: List of closest points (one on each line), shape (n, 3)
    """
    
    n = len(points)  # Number of lines
    points = np.array(points)
    directions = np.array(directions)
    
    # Initialize matrix A and vector b for solving the system
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Fill in the matrix A and vector b based on the vector differences
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = np.dot(directions[i], directions[i])  # d_i . d_i
            else:
                A[i, j] = -np.dot(directions[i], directions[j])  # -d_i . d_j
        
        # Compute the right-hand side vector b_i
        for k in range(n):
            if i != k:
                p_ik = points[k] - points[i]
                b[i] += np.dot(directions[i], p_ik)
    
    # Solve the system for t1, t2, ..., tn
    t_values = np.linalg.solve(A, b)
    
    # Compute the closest points on each line
    closest_points = []
    for i in range(n):
        closest_point = points[i] + t_values[i] * directions[i]
        closest_points.append(closest_point)
    
    closest_points = np.array(closest_points)
    return closest_points