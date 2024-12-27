import os
import json
import argparse
import cv2  # 用于读取图像尺寸


def yolo_to_labelme(yolo_file, image_file, output_file):
    """
    将 YOLO 格式标注文件转换为 Labelme JSON 格式
    :param yolo_file: YOLO 格式标注文件路径
    :param image_file: 图像文件路径
    :param output_file: 输出 JSON 文件路径
    """
    # 使用 OpenCV 读取图像的宽度和高度
    image = cv2.imread(image_file)
    if image is None:
        print(f"无法读取图像文件：{image_file}")
        return
    image_height, image_width, _ = image.shape

    # 初始化 Labelme 格式 JSON 数据结构
    labelme_data = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_file),
        "imageHeight": image_height,
        "imageWidth": image_width,
        "imageData": None,
    }

    # 读取 YOLO 标注文件
    with open(yolo_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 13:
            print(f"跳过无效的标注行：{line}")
            continue

        # 解析类别和边界框信息
        label = parts[0]
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        # 计算边界框的顶点坐标（矩形框的左上角和右下角）
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # 添加矩形框到 JSON 数据
        labelme_data["shapes"].append({
            "label": f"bbox_{label}",
            "points": [[x1, y1], [x2, y2]],  # 左上角和右下角的坐标
            "group_id": None,
            "shape_type": "rectangle",  # 矩形框
            "flags": {}
        })

        # 解析关键点
        keypoints = [
            (float(parts[5]) * image_width, float(parts[6]) * image_height),  # pt1 (左上)
            (float(parts[7]) * image_width, float(parts[8]) * image_height),  # pt2 (右上)
            (float(parts[9]) * image_width, float(parts[10]) * image_height),  # pt3 (右下)
            (float(parts[11]) * image_width, float(parts[12]) * image_height)  # pt4 (左下)
        ]

        # 将关键点添加到 Labelme 的 shapes 中
        for idx, (kx, ky) in enumerate(keypoints):
            if kx == -1 or ky == -1:
                continue  # 跳过无效的关键点
            labelme_data["shapes"].append({
                "label": f"keypoint_{label}_{idx + 1}",
                "points": [[kx, ky]],  # 点的坐标
                "group_id": None,
                "shape_type": "point",  # 点
                "flags": {}
            })

    # 保存为 Labelme JSON 文件
    with open(output_file, "w") as json_file:
        json.dump(labelme_data, json_file, indent=4)

    print(f"转换完成，保存到：{output_file}")


def process_folder(yolo_folder, image_folder, output_folder):
    """
    批量将 YOLO 标注文件转换为 Labelme JSON 格式
    :param yolo_folder: YOLO 标注文件夹路径
    :param image_folder: 图像文件夹路径
    :param output_folder: 输出 JSON 文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    yolo_files = [f for f in os.listdir(yolo_folder) if f.endswith(".txt")]
    for yolo_file in yolo_files:
        yolo_path = os.path.join(yolo_folder, yolo_file)
        image_name = os.path.splitext(yolo_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"图像文件未找到，跳过：{image_path}")
            continue

        output_file = os.path.join(output_folder, os.path.splitext(yolo_file)[0] + ".json")
        yolo_to_labelme(yolo_path, image_path, output_file)


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="将 YOLO 带关键点的标注文件转换为 Labelme JSON 格式")
    parser.add_argument("--yolo_folder", required=True, help="YOLO 格式标注文件夹路径")
    parser.add_argument("--image_folder", required=True, help="图像文件夹路径")
    parser.add_argument("--output_folder", required=True, help="输出 JSON 文件夹路径")

    args = parser.parse_args()

    # 批量转换
    process_folder(args.yolo_folder, args.image_folder, args.output_folder)
