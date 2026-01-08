import cv2
import numpy as np
import open3d as o3d
import os
import json

def process_depth_image_for_obstacles(image_path):
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

    # Extract the obstacles mask
    obstacle_mask = np.zeros_like(depth_image, dtype=np.uint8)
    outlier_points = np.asarray(outlier_cloud.points)
    for i in range(outlier_points.shape[0]):
        u = int((outlier_points[i][0] * fx) / outlier_points[i][2] + cx)
        v = int((outlier_points[i][1] * fy) / outlier_points[i][2] + cy)
        if 0 <= u < width and 0 <= v < height:
            obstacle_mask[v, u] = 255

    # Remove small noise (optional)
    cleaned_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    return cleaned_mask

def draw_bounding_boxes(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    bounding_boxes_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing colored boxes
    bounding_boxes = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bounding_boxes_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green bounding box
        bounding_boxes.append({
            "category": "obstacle",
            "id": idx,
            "attributes": {
                "occluded": False,
                "truncated": False,
                "trafficLightColor": "none"
            },
            "box2d": {
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + w),
                "y2": float(y + h)
            }
        })

    return bounding_boxes_image, bounding_boxes

if __name__ == "__main__":
    input_folder = '/home/naren/final_ws/final_dataset/depth_combined/'  # Update this path as needed
    output_folder = '/home/naren/final_ws/final_dataset/depth_obstacle/'  # Update this path as needed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1, 39):  # Adjust the range as needed
        image_filename = f'{i}_depth.png'
        image_path = os.path.join(input_folder, image_filename)
        try:
            cleaned_mask = process_depth_image_for_obstacles(image_path)
            bounding_boxes_image, bounding_boxes = draw_bounding_boxes(cleaned_mask)
            
            output_filename_mask = f'{i}.png'
            output_filename_bboxes = f'{i}_bbox.png'
            output_filename_json = f'{i}.json'
            output_path_mask = os.path.join(output_folder, output_filename_mask)
            output_path_bboxes = os.path.join(output_folder, output_filename_bboxes)
            output_path_json = os.path.join(output_folder, output_filename_json)
            
            cv2.imwrite(output_path_mask, cleaned_mask)  # Save the cleaned mask
            cv2.imwrite(output_path_bboxes, bounding_boxes_image)  # Save the image with bounding boxes
            
            with open(output_path_json, 'w') as json_file:
                json.dump({"objects": bounding_boxes}, json_file, indent=4)  # Save the bounding boxes in JSON format
            
            print(f"Obstacle mask saved to {output_path_mask}")
            print(f"Bounding boxes image saved to {output_path_bboxes}")
            print(f"Bounding boxes JSON saved to {output_path_json}")
        except Exception as e:
            print(f"An error occurred while processing {image_filename}: {e}")
