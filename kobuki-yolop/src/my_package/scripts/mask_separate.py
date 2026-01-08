import cv2
import numpy as np
import open3d as o3d
import os

def process_depth_image(image_path):
    # Load the depth image
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # Ensure the depth image is single-channel
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    elif len(depth_image.shape) != 2:
        raise ValueError("Invalid depth image format")

    # Normalize the depth image
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convert depth image to point cloud
    height, width = depth_image.shape
    fx, fy = 500.0, 500.0  # Focal lengths (example values, adjust as needed)
    cx, cy = width / 2, height / 2  # Principal point (center of image)
    
    points = []
    for v in range(height):
        for u in range(width):
            z = depth_image_normalized[v, u]
            if z > 0:  # Ignore zero depth values
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    
    # Check if points are available
    if not points:
        raise ValueError("No valid points found in the depth image")

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

    # Perform RANSAC to segment the ground plane
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    if len(inliers) == 0:
        raise ValueError("RANSAC did not find a plane")

    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Extract the ground plane mask
    ground_mask = np.zeros_like(depth_image, dtype=np.uint8)
    inlier_points = np.asarray(inlier_cloud.points)
    for i in range(inlier_points.shape[0]):
        u = int((inlier_points[i][0] * fx) / inlier_points[i][2] + cx)
        v = int((inlier_points[i][1] * fy) / inlier_points[i][2] + cy)
        if 0 <= u < width and 0 <= v < height:
            ground_mask[v, u] = 255

    # Remove obstacles (keep ground plane only)
    cleaned_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return cleaned_mask

if __name__ == "__main__":
    input_folder = '/home/naren/final_ws/final_dataset/depth_combined/'  # Update this path as needed
    output_folder = '/home/naren/final_ws/final_dataset/depth_ground/'  # Update this path as needed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(24, 39):  # Adjust the range as needed
        image_filename = f'{i}_depth.png'
        image_path = os.path.join(input_folder, image_filename)
        try:
            cleaned_mask = process_depth_image(image_path)
            output_filename = f'{i}.png'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cleaned_mask)  # Save the cleaned mask
            print(f"Cleaned mask saved to {output_path}")
        except Exception as e:
            print(f"An error occurred while processing {image_filename}: {e}")
