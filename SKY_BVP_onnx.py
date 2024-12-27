import onnxruntime
import numpy as np
import cv2
import os
import argparse
import time
import copy

def my_letter_box(img, size=(640, 640)):  
    h, w, c = img.shape
    r = min(size[0]/h, size[1]/w)
    new_h, new_w = int(h * r), int(w * r)
    top = (size[0] - new_h) // 2
    left = (size[1] - new_w) // 2
    
    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, r, left, top

def xywh2xyxy(boxes):   
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh

def my_nms(boxes, iou_thresh):  
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
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(boxes, r, left, top):  
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    return boxes

def restore_landmarks(landmarks, r, left, top):  # Restore landmarks coordinates
    landmarks[:, 0] -= left
    landmarks[:, 1] -= top
    landmarks[:, 0] /= r
    landmarks[:, 1] /= r
    return landmarks

def detect_pre_processing(img, img_size):  
    img_padded, r, left, top = my_letter_box(img, img_size)
    img = img_padded[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)  # BGR to RGB and transpose
    img = img / 255.0  # Normalize to 0-1 range
    img = img.reshape(1, *img.shape)  # Add batch dimension
    return img, r, left, top

def post_processing(dets, r, left, top, conf_thresh=0.2, iou_thresh=0.5):
    choice = dets[:, :, 4] > conf_thresh  # Confidence thresholding
    dets = dets[choice]
    dets[:, 13:15] *= dets[:, 4:5]  # Adjust landmark scores by confidence
    box = dets[:, :4]
    boxes = xywh2xyxy(box)  # Convert to x1, y1, x2, y2 format
    score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
    output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)

    keep = my_nms(output, iou_thresh)  # Apply NMS
    output = output[keep]

    if len(output) == 0:
        print("No plates detected.")
        return []

    output = restore_box(output, r, left, top)

    # Restore landmark coordinates as well (assuming the model outputs 4 keypoints)
    for i in range(len(output)):
        output[i, 5:13] = restore_landmarks(output[i, 5:13].reshape(-1, 2), r, left, top).flatten()

    return output

def allFilePath(rootPath,allFIleList):  
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

def draw_result(org_img, output):
    if len(output) == 0:  
        print("No plates detected.")
        return org_img

    for result in output:
        rect = result[:4].astype(int)
        landmarks = result[5:13].reshape(-1, 2).astype(int)  # 4 keypoints
        cv2.rectangle(org_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), 2)  # Draw the bounding box
        for lm in landmarks:
            cv2.circle(org_img, tuple(lm), 5, (0, 255, 0), -1)  # Draw keypoints (4 points)
    return org_img

if __name__ == "__main__":
    begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', type=str, default='', help='model.pt path(s)')  # 检测模型
    parser.add_argument('--image_path', type=str, default='', help='source') 
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='', help='source') 
    opt = parser.parse_args()

    # Get image paths
    image_files = []
    allFilePath(opt.image_path, image_files)

    providers = ['CPUExecutionProvider']  # Use CPU for inference
    img_size = (opt.img_size, opt.img_size)

    # Load ONNX model
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers)

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    save_path = opt.output

    count = 0
    for image_path in image_files:
        count += 1
        print(f"Processing {count}: {image_path}")
        img = cv2.imread(image_path)
        img0 = copy.deepcopy(img)
        img, r, left, top = detect_pre_processing(img, img_size)

        # Run detection on ONNX model
        y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
        output = post_processing(y_onnx, r, left, top)

        # Draw and save result
        img_with_result = draw_result(img0, output)
        img_name = os.path.basename(image_path)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, img_with_result)

    print(f"Total time: {time.time() - begin} s")
