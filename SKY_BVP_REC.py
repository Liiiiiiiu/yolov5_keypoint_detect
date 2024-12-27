import os
import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
import time
import onnxruntime
import argparse
from alphabets import plate_chr

# Function to handle image paths with non-ASCII characters
def cv_imread(path):
    """
    Read an image with support for non-ASCII paths.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

mean_value, std_value = (0.588, 0.193)  # Mean and standard deviation for normalization

# Decode the plate number from the model's predictions
def decodePlate(preds):
    """
    Decode the predicted indices to get the plate number string.
    """
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return plate

# Pre-process the input image for the recognition model
def rec_pre_precessing(img, size=(48, 168)):
    """
    Pre-process the input image for the recognition model by resizing, normalizing, and reshaping.
    """
    print(f"Input image shape: {img.shape}")  # Print the shape of the input image
    img = cv2.resize(img, (168, 48))  # Resize image to the target size
    img = img.astype(np.float32)
    img = (img / 255 - mean_value) / std_value  # Normalize with mean and std
    img = img.transpose(2, 0, 1)  # Convert (H, W, C) to (C, H, W)
    img = img.reshape(1, *img.shape)  # Add batch dimension
    print(f"Processed image shape: {img.shape}")  # Print the processed image shape
    return img

# Get the license plate result using the recognition model
def get_plate_result(img, session_rec):
    """
    Perform license plate recognition and return the plate number.
    """
    img = rec_pre_precessing(img)  # Preprocess the image
    y_onnx = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: img})[0]
    
    index = np.argmax(y_onnx[0], axis=1)  # Find the class index with max probability
    plate_no = decodePlate(index)  # Decode the plate number
    print(f"Decoded plate number: {plate_no}")  # Print the decoded plate number
    return plate_no

# Load the detection model
def load_model(weights, device):
    """
    Load the detection model from the specified weights file.
    """
    model = attempt_load(weights, map_location=device)
    return model

# Pre-process the input image for the detection model
def preprocess_image(img, img_size):
    """
    Pre-process the input image for the detection model by resizing and normalizing.
    """
    img = letterbox(img, new_shape=img_size)[0]  # Resize while maintaining aspect ratio
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # Convert BGR to RGB and rearrange dimensions
    img = torch.from_numpy(img).float() / 255.0  # Normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    return img

# Scale coordinates and landmarks back to the original image
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

# Perform perspective transformation to rectify the plate
def four_point_transform(image, pts):
    """
    Perform a perspective transformation to rectify the plate image.
    """
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

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Process all images in the given folder
def process_images(image_folder, output_folder, model, device, img_size, session_rec):
    """
    Process all images in the input folder, detect plates, and save results in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in Path(image_folder).rglob("*.[jp][pn]g"):
        print(f"Processing: {img_file}")
        
        img = cv_imread(str(img_file))
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue

        detections, landmarks, class_id = detect_plates_yolo(model, img, device, img_size)
        result_img = img.copy()

        for i, pts in enumerate(landmarks):
            if class_id[i] == 2:
                cropped_plate = img[int(pts[1]):int(pts[3]), int(pts[0]):int(pts[2])]
                plate_number = get_plate_result(cropped_plate, session_rec)

                save_yolo_format(
                    [detections[i]], [landmarks[i]], [class_id[i]], img.shape,
                    os.path.join(output_folder, f"{img_file.stem}.txt")
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to input images folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder")
    parser.add_argument("--weights", type=str, required=True, help="Path to detection model weights")
    parser.add_argument("--rec_weights", type=str, required=True, help="Path to recognition model weights")
    parser.add_argument("--img_size", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    session_rec = onnxruntime.InferenceSession(args.rec_weights, providers=['CPUExecutionProvider'])

    process_images(args.image_folder, args.output_folder, model, device, (args.img_size, args.img_size), session_rec)
