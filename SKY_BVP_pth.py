import os
import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
import time

def preprocess_image(img, img_size):
    """
    Resize the image to the specified size while maintaining the aspect ratio.
    """
    img = letterbox(img, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # Convert BGR to RGB and adjust dimensions
    img = torch.from_numpy(img).float() / 255.0  # Normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Scale detection coordinates and landmarks back to the original image size.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # Scale gain
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # Padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # Adjust x-coordinates
    coords[:, [1, 3, 5, 7]] -= pad[1]  # Adjust y-coordinates
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords

def four_point_transform(image, pts):
    """
    Apply perspective transformation to straighten the license plate image.
    :param image: Input image
    :param pts: Four corner points of the license plate
    :return: Transformed license plate image
    """
    # Calculate the perspective transformation matrix
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    start_time = time.time()
    
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    end_time = time.time()
    execution_time = end_time - start_time
    return warped

def draw_label_and_bbox(image, detections, landmarks, save_path, class_ids):
    """
    Draw detection boxes, class information, and landmarks on the image, and save to the specified path.
    :param image: Input image
    :param detections: Detection boxes [x_min, y_min, x_max, y_max]
    :param landmarks: Corner points [[pt1x, pt1y], [pt2x, pt2y], [pt3x, pt3y], [pt4x, pt4y]]
    :param save_path: Path to save the result image
    :param class_ids: Class IDs for each detection box
    """
    class_mapping = {0: "body", 1: "vehicle", 2: "plate"}
    class_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # Blue, green, red

    for i, det in enumerate(detections):
        x_min, y_min, x_max, y_max = map(int, det[:4])
        class_id = int(class_ids[i])
        label = class_mapping.get(class_id, "unknown")
        color = class_colors.get(class_id, (255, 255, 255))  # Default color is white
        
        # Draw the detection box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw class label above the box
        font_scale = 1.0
        font_thickness = 2
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Draw landmarks only for class 2 (plate)
        if class_id == 2:
            for j, (x, y) in enumerate(landmarks[i]):
                x, y = int(x), int(y)
                cv2.circle(image, (x, y), 5, color, -1)  # Mark the landmark with a small circle

    # Save the result image
    cv2.imwrite(save_path, image)
    print(f"Result saved to: {save_path}")

def detect_plates_yolo(model, img, device, img_size, conf_thres=0.3, iou_thres=0.5):
    """
    Perform plate detection using the YOLO model.
    """
    img_original = img.copy()
    img = preprocess_image(img, img_size).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    
    # Non-maximum suppression
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    detections = []
    landmarks = []
    class_id = []

    # Process detection results
    for det in pred:
        if det is not None and len(det):
            # Scale coordinates back to the original image
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_original.shape).round()
            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], img_original.shape).round()

            detections.extend(det[:, :4].tolist())
            landmarks.extend(det[:, 5:13].view(-1, 4, 2).tolist())
            class_id.extend(det[:, -1].tolist())
    
    return detections, landmarks, class_id

def load_model(weights, device):
    """
    Load the YOLO model from the given weights.
    """
    model = attempt_load(weights, map_location=device)
    return model

def process_images(image_folder, output_folder, model, device, img_size):
    """
    Process all images in the input folder, perform detection, and save results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the labels folder
    annotations_folder = os.path.join(output_folder, "labels")
    os.makedirs(annotations_folder, exist_ok=True)

    # Loop through all images in the input folder
    for img_file in Path(image_folder).rglob("*.[jp][pn]g"):  # Handle .jpg and .png formats
        print(f"Processing: {img_file}")
        
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue

        base_name = img_file.stem  # Get the filename without extension
        detections, landmarks, class_id = detect_plates_yolo(model, img, device, img_size)

        # Save YOLO-format annotations
        save_annotations(str(img_file), zip(detections, landmarks, class_id), annotations_folder)

        # Save detection results with annotations
        result_img_path = os.path.join(output_folder, f"results/{base_name}_result.jpg")
        os.makedirs(os.path.dirname(result_img_path), exist_ok=True)
        draw_label_and_bbox(img, detections, landmarks, result_img_path, class_id)

    print("Processing completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, required=True, help='Path to YOLO detection model')
    parser.add_argument('--image_path', type=str, required=True, help='Folder containing input images')
    parser.add_argument('--output', type=str, required=True, help='Folder to save YOLO-format labels and results')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size for YOLO model')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = load_model(opt.detect_model, device)

    process_images(opt.image_path, opt.output, detect_model, device, opt.img_size)
