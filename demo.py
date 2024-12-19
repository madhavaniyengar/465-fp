#!/usr/bin/env python
import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def get_transform(p, q):
    p_centered = p - np.mean(p, axis=0)
    q_centered = q - np.mean(q, axis=0)
    S = p_centered.T @ q_centered
    u, s, vt = np.linalg.svd(S)
    mat = np.eye(3)
    mat[2, 2] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ mat @ u.T
    t = np.mean(q, axis=0) - np.mean(p, axis=0) @ R.T
    t = np.expand_dims(t, 1)
    return R, t

def compute_normals(k_points):
    u, s, vt = np.linalg.svd(k_points.T @ k_points)
    
    # get normal from eigenvector with smallest eigenvalue
    normal = vt[-1]
    normal = np.expand_dims(normal, 1)
    
    return normal

def compute_k_neighbors(points, k):
    k_neighbors_list = []
    for point in points:
        # get k nearest neighbors
        distances = np.linalg.norm(points - point, axis=1)
        k_neighbors = points[np.argsort(distances)[:k]]
        k_neighbors_list.append(k_neighbors)
    return np.array(k_neighbors_list)
        

def compute_point_normals(k_neighbors_list):
    normals = []
    for k_neighbors in k_neighbors_list:
        normal = compute_normals(k_neighbors)
        # print('normal shape', normal.shape)
        normals.append(normal)
    return np.array(normals)
        
def get_point_features(points, k_neighbors_list, normals):
    point_features = []
    for point_idx, point in enumerate(points):
        k_neighbors = k_neighbors_list[point_idx]
        normal = normals[point_idx]
        point_feature = []
        for t, pt in enumerate(k_neighbors):
            for s, ps in enumerate(k_neighbors):
                if t == s:
                    continue
                d = np.linalg.norm(pt - ps)
                u = normals[s]
                v = np.cross(u, (pt - ps)/d)
                w = np.cross(u, v)
                alpha = np.dot(v, normals[t])
                phi = np.dot(u, (pt-ps)/d)
                theta = np.arctan2(np.dot(w, normals[t]), np.dot(u, normals[t]))
                point_feature.append([alpha, phi, theta, d])
        point_features.append(point_feature)
    return np.array(point_features)

def get_point_feature_histogram(point_features, bins):
    histograms = []
    for point_feature in point_features:
        histogram = np.histogramdd(point_feature, bins=bins, density=False)[0]
        # histogram = histogram / np.linalg.norm(histogram, ord='nuc')
        histograms.append(histogram)
    return np.array(histograms)
    
def get_point_signature(histograms):
    signatures = []
    for histogram in histograms:
        signatures.append(histogram.flatten() / np.linalg.norm(histogram.flatten(), ord=1))
    return np.array(signatures)
            

def icp_pfh(pcd_source, pcd_target, num_neighbors, bins, save_file):
    #Import the cloud
    pc_source = utils.load_pc(pcd_source)

    ###YOUR CODE HERE###
    pc_target = utils.load_pc(pcd_target) # Change this to load in a different target

    errors = []
    pc_source = np.array(pc_source).squeeze()
    pc_target = np.array(pc_target).squeeze()
    cp = None
    cq = None
    i = 0
    start_time = time.time()
    while i < 3:
        correspondences = []
        # compute signatures for each point
        # print('pc_source shape', pc_source.shape)
        k_neighbors_list = compute_k_neighbors(pc_source, num_neighbors)
        # print('k neighbors list shape', k_neighbors_list.shape)
        normals = compute_point_normals(k_neighbors_list).squeeze()
        # print('normals shape', normals.shape)
        point_features = get_point_features(pc_source, k_neighbors_list, normals)
        # print('point features shape', point_features.shape)
        histograms = get_point_feature_histogram(point_features, bins)
        # print('histograms shape', histograms.shape)
        signatures = get_point_signature(histograms) # a signature vector for each point
        # print('signatures shape', signatures.shape)
        # print('signatures 0', signatures[0])
        # print('signatures 1', signatures[1])
        # print('signatures shape', signatures.shape)
        
        for point_idx, point in enumerate(pc_source):
            # find the closest point in pc_target
            min_distance = np.inf
            closest_point = None
            for target_idx, target in enumerate(pc_target):
                d = np.linalg.norm(signatures[target_idx] - signatures[point_idx])
                # print('d', d)
                if d < min_distance:
                    min_distance = d
                    closest_point = target
            correspondences.append((point, closest_point))
        correspondences = np.array(correspondences)
        cp = correspondences[:, 0]
        cq = correspondences[:, 1]
        # print('cp shape', cp.shape)
        # print('cq shape', cq.shape)
        R, t = get_transform(cp, cq)
        curr_error = np.linalg.norm(R @ cp.T + t - cq.T) ** 2
        # if len(errors) > 0 and curr_error > errors[-1]:
        #     break
        errors.append(curr_error)
        # print('R', R)
        # print('t', t)
        pc_source = ((R @ pc_source.T) + t).T
        # if np.linalg.norm(R @ cp.T + t - cq.T) ** 2 < epsilon:
        #     break
        i += 1
    
    # print('Time taken for ICP w/ PFH:', time.time() - start_time, 'seconds')
    pc_source = np.expand_dims(pc_source, 2)
    pc_target = np.expand_dims(pc_target, 2)
    
    # plot of error per iteration
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    plt.title('Error per iteration (PFH distance metric)')
    plt.savefig(f'{save_file}/errors_pfh.png')

    # print(pc_source)
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title('Point clouds aligned using ICP with PFH signature distance metric')
    plt.savefig(f'{save_file}/pcds_pfh.png')


def icp(pcd_source, pcd_target, num_neighbors, bins, save_file):
    pc_source = utils.load_pc(pcd_source)
    pc_target = utils.load_pc(pcd_target)
    errors = []
    pc_source = np.array(pc_source).squeeze()
    pc_target = np.array(pc_target).squeeze()
    cp = None
    cq = None
    start_time = time.time()
    i = 0
    while i < 30:
        correspondences = []
        for point in pc_source:
            # find the closest point in pc_target
            min_distance = np.inf
            closest_point = None
            for target in pc_target:
                d = np.linalg.norm(target - point)
                if d < min_distance:
                    min_distance = d
                    closest_point = target
            correspondences.append((point, closest_point))
        correspondences = np.array(correspondences)
        cp = correspondences[:, 0]
        cq = correspondences[:, 1]
        R, t = get_transform(cp, cq)
        errors.append(np.linalg.norm(R @ cp.T + t - cq.T) ** 2)
        pc_source = ((R @ pc_source.T) + t).T
        i += 1
    
    # print('Time taken for ICP w/ Euclidean distance:', time.time() - start_time, 'seconds')
    
    pc_source = np.expand_dims(pc_source, 2)
    pc_target = np.expand_dims(pc_target, 2)
    
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    plt.title('Error per iteration (Euclidean distance metric)')
    plt.savefig(f'{save_file}/errors_euclidean.png')

    # print(pc_source)
    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.title('Point clouds aligned using ICP with Euclidean distance metric')
    plt.savefig(f'{save_file}/pcds_euclidean.png')
    

def main(pcd_source, pcd_target):
    print('running demo.py: Comparing ICP with Euclidean distance metric and ICP with PFH signature distance metric')
    print('Expected time for script to run: < 1 minute')
    save_path = f'output/'
    os.makedirs(save_path, exist_ok=True)
    
    # plot the starting pose of the source and target point clouds
    source = utils.load_pc(pcd_source)
    target = utils.load_pc(pcd_target)
    utils.view_pc([source, target], None, ['b', 'r'], ['o', '^'])
    plt.title('Unaligned point clouds')
    plt.savefig(f'{save_path}/unalined_pcds.png')
    
    icp(pcd_source, pcd_target, 5, 5, f'{save_path}')
    icp_pfh(pcd_source, pcd_target, 5, 5, f'{save_path}')
    print('Check the output/ folder for the results!')

if __name__ == '__main__':
    main('cloud_icp_source.csv', 'cloud_icp_target3.csv')
