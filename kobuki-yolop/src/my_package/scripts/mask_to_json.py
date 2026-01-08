import json
import os
import cv2

def create_annotation(mask_path, category="area/drivable", object_id=7):
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image at {mask_path}")

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the mask image")

    # Assuming the largest contour is the object
    contour = max(contours, key=cv2.contourArea)

    # Create the poly2d structure
    poly2d = []
    for point in contour:
        x, y = point[0]
        poly2d.append([float(x), float(y), "L"])  # Use "L" for line vertices

    # Close the polygon by adding the first point at the end
    poly2d.append([float(contour[0][0][0]), float(contour[0][0][1]), "L"])

    # Create the annotation dictionary
    annotation = {
        "category": category,
        "id": object_id,
        "attributes": {},
        "poly2d": poly2d
    }

    return annotation

if __name__ == "__main__":
    input_folder = '/home/naren/final_ws/final_dataset/depth_ground/'  # Update this path as needed
    output_folder = '/home/naren/final_ws/final_dataset/depth_ground/'  # Specify the output folder for annotations

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(24, 39):  # Adjust the range as needed
        image_filename = f'{i}.png'
        image_path = os.path.join(input_folder, image_filename)
        json_filename = f'{i}.json'
        json_path = os.path.join(output_folder, json_filename)
        
        try:
            annotation = create_annotation(image_path, object_id=7)
            
            # Check if the JSON file already exists and load existing data
            existing_data = {}
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            
            # Add the new annotation under 'objects'
            if 'objects' not in existing_data:
                existing_data['objects'] = []
            existing_data['objects'].append(annotation)
            
            # Save updated JSON file
            with open(json_path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
                
            print(f"Annotation for {image_filename} saved to {json_filename}")
        except Exception as e:
            print(f"An error occurred while processing {image_filename}: {e}")
