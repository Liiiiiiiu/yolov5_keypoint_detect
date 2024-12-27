import os
import cv2
import numpy as np
import onnxruntime
import argparse
import time
import copy
from tqdm import tqdm  # Import tqdm for progress bar
import shutil

# --------------------- Image Processing Module ---------------------

def my_letter_box(img, size=(640, 640)):  
    """
    Perform image padding to maintain the aspect ratio and resize the image to a specified size.
    """
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = (size[0] - new_h) // 2
    left = (size[1] - new_w) // 2
    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, r, left, top

def restore_box(boxes, r, left, top):  
    """
    Restore bounding box positions (align with the original image size).
    """
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    return boxes

def restore_landmarks(landmarks, r, left, top):  
    """
    Restore landmark positions (align with the original image size).
    """
    landmarks[:, 0] -= left
    landmarks[:, 1] -= top
    landmarks[:, 0] /= r
    landmarks[:, 1] /= r
    return landmarks

# --------------------- Non-Maximum Suppression (NMS) Module ---------------------

def my_nms(boxes, iou_thresh):  
    """
    Perform Non-Maximum Suppression (NMS).
    """
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + \
                     (boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

# --------------------- Inference Module ---------------------

def detect_pre_processing(img, img_size):  
    """
    Preprocess the image for inference, including padding and normalization.
    """
    img_padded, r, left, top = my_letter_box(img, img_size)
    img = img_padded[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)  # Convert BGR to RGB and transpose
    img = img / 255.0  # Normalize to the range [0, 1]
    img = img.reshape(1, *img.shape)  # Add batch dimension
    return img, r, left, top

def post_processing(dets, r, left, top, conf_thresh=0.2, iou_thresh=0.5):
    """
    Post-process detection results, including confidence thresholding and NMS.
    """
    choice = dets[:, :, 4] > conf_thresh  # Filter by confidence threshold
    dets = dets[choice]
    dets[:, 13:] *= dets[:, 4:5]  # Adjust landmark scores by confidence
    box = dets[:, :4]
    boxes = xywh2xyxy(box)  # Convert to x1, y1, x2, y2 format
    score = np.max(dets[:, 13:], axis=-1, keepdims=True)
    
    index = np.argmax(dets[:, 13:], axis=1).reshape(-1, 1)
    key = dets[:, 5:13]
    
    output = np.concatenate((boxes, score, key, index), axis=1)
    keep = my_nms(output, iou_thresh)  # Apply NMS
    output = output[keep]

    if len(output) == 0:
        return []

    output = restore_box(output, r, left, top)
    
    for i in range(len(output)):
        output[i, 5:13] = restore_landmarks(output[i, 5:13].reshape(-1, 2), r, left, top).flatten()

    return output

def allFilePath(rootPath, allFileList):  
    """
    Traverse the folder to get all file paths.
    """
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFileList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFileList)

def xywh2xyxy(boxes):   
    """
    Convert bounding boxes from xywh format to xyxy format.
    """
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh

def save_yolo_format(output, img_shape, save_path):
    """
    Save the detection results in YOLO format. License plate categories include landmark information, while other categories use -1 for keypoints.
    """
    h, w, _ = img_shape
    yolo_labels = []
    
    for result in output:
        class_id = int(result[-1])
        box = result[:4]
        
        # Normalize bounding boxes
        x_center = (box[0] + box[2]) / 2 / w
        y_center = (box[1] + box[3]) / 2 / h
        width = (box[2] - box[0]) / w
        height = (box[3] - box[1]) / h
        
        # Process keypoints
        if class_id == 2:  # License plates (class 2) save keypoints
            keypoints = result[5:13].reshape(-1, 2)
            keypoints = (keypoints / np.array([w, h])).flatten()  # Normalize keypoints
        else:  # For classes 0 and 1, fill keypoints with -1
            keypoints = [-1] * 8  # Use -1 to indicate no keypoints
        
        # Generate YOLO label
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height} " + " ".join(map(str, keypoints)))
    
    # Ensure that 'labels' folder exists in the save path
    dir_path, file_name = os.path.split(save_path)
    new_save_dir = os.path.join(dir_path, 'labels')
    os.makedirs(new_save_dir, exist_ok=True)

    new_save_path = os.path.join(new_save_dir, os.path.splitext(file_name)[0] + '.txt')
    
    # Save YOLO labels to a file
    with open(new_save_path, 'w') as f:
        f.write("\n".join(yolo_labels))

def draw_result(org_img, output, save_path, original_img_path, save_original=False):
    """
    Draw results and save YOLO-format annotations. Only save annotations for category 2 and optionally save the original image.
    """
    detected_plate = False  # Flag to indicate if a license plate (class 2) is detected

    for result in output:
        class_id = int(result[-1])
        rect = result[:4].astype(int)
        landmarks = result[5:13].reshape(-1, 2).astype(int)  # Four keypoints
        cv2.rectangle(org_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), 2)  # Draw rectangle
        cv2.putText(org_img, f'Class ID: {class_id}', (rect[0], rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if class_id == 2:  # If it is a license plate, draw keypoints
            detected_plate = True
            for landmark in landmarks:
                cv2.circle(org_img, tuple(landmark), 5, (0, 0, 255), -1)  # Draw keypoints

    # If a license plate (class 2) is detected, save the annotation file and image
    if detected_plate:
        save_yolo_format(output, org_img.shape, save_path)
        
        result_folder = os.path.join(os.path.dirname(save_path), 'results')
        os.makedirs(result_folder, exist_ok=True)
        
        save_img_path = os.path.join(result_folder, os.path.basename(save_path))
        cv2.imwrite(save_img_path, org_img)
        
        if save_original:
            # Copy the original image to the target folder
            original_img_name = os.path.basename(original_img_path)
            save = os.path.dirname(save_path)
            save_original_path = os.path.join(save, "original", original_img_name)
            os.makedirs(os.path.dirname(save_original_path), exist_ok=True)
            shutil.copy(original_img_path, save_original_path)

    return org_img

# --------------------- Main Program ---------------------

def load_model(model_path, providers=['CPUExecutionProvider']):
    """
    Load the ONNX model.
    """
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    return session

def run_inference(model, img):
    """
    Run inference on the input image.
    """
    result = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: img})[0]
    return result

# Main function to process all images
def main():
    start_time = time.time()

    # Set command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='', help='Path to detection model (.onnx)')
    parser.add_argument('--image_path', type=str, default='', help='Path to the input image folder') 
    parser.add_argument('--img_size', type=int, default=640, help='Inference image size (pixels)')
    parser.add_argument('--output', type=str, default='', help='Path to save output results') 
    opt = parser.parse_args()

    # Get all image paths
    image_files = []
    allFilePath(opt.image_path, image_files)

    # Load the detection model
    session_detect = load_model(opt.detect_model)

    # Create the output folder if it doesn't exist
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    save_path = opt.output

    # Initialize class counters
    class_counts = {0: 0, 1: 0, 2: 0}  # 0: human, 1: vehicle, 2: license plate

    # Process each image
    for count, image_path in enumerate(tqdm(image_files, desc="Processing images")):  # Use tqdm to display progress bar
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Unable to load image: {image_path}")
            continue  # Skip current image and process the next one
        img0 = img.copy()  # Original image for result visualization
        img, r, left, top = detect_pre_processing(img, (opt.img_size, opt.img_size))

        # Run inference
        y_onnx = run_inference(session_detect, img)

        # Post-process and draw results
        output = post_processing(y_onnx, r, left, top)
        img_with_result = draw_result(img0, output, os.path.join(save_path, os.path.basename(image_path)), image_path, save_original=True)

        # Update class counters
        for result in output:
            class_id = int(result[-1])
            if class_id in class_counts:
                class_counts[class_id] += 1

    # Print detection statistics
    print("\n[INFO] Detection statistics:")
    for class_id, count in class_counts.items():
        print(f"  Class {class_id}: {count} objects detected")

    print(f"[INFO] Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
