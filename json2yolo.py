import os
import json
import argparse
import cv2  # 用于读取图像尺寸


def labelme_to_yolo(json_file, image_width, image_height, output_file):
    """
    将 Labelme JSON 格式转换为 YOLO 格式
    :param json_file: Labelme JSON 文件路径
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :param output_file: 输出 YOLO 格式文件路径
    """
    # 读取 JSON 文件
    with open(json_file, "r") as f:
        labelme_data = json.load(f)

    yolo_lines = []  # 保存 YOLO 格式的标注行
    shapes = labelme_data.get("shapes", [])

    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        shape_type = shape["shape_type"]

        # 初始化关键点，默认值为 -1
        keypoints = [-1, -1, -1, -1, -1, -1, -1, -1]

        # 如果是矩形框
        if shape_type == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 转为 YOLO 中心点坐标、宽高的归一化格式
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height

            # 提取类别编号（从 `bbox_0`, `bbox_1`, `bbox_2` 中动态获取）
            if "bbox_" in label:
                class_id = int(label.split("_")[1])  # 提取类别编号
            else:
                print(f"未知标签格式：{label}，跳过此条")
                continue

            # 检查类别，类别 2 提取关键点，其余类别关键点默认保留 -1
            if class_id == 2:  # 仅类别 2 有关键点
                for keypoint_shape in shapes:
                    if keypoint_shape["shape_type"] == "point" and keypoint_shape["label"].startswith("keypoint_2"):
                        keypoint_index = int(keypoint_shape["label"].split("_")[2]) - 1  # 获取关键点索引（0-3）
                        kx, ky = keypoint_shape["points"][0]
                        keypoints[keypoint_index * 2] = kx / image_width
                        keypoints[keypoint_index * 2 + 1] = ky / image_height

            # 将矩形框和关键点存入 YOLO 格式
            yolo_line = f"{class_id} {x_center} {y_center} {width} {height} "
            yolo_line += " ".join(map(str, keypoints))  # 添加关键点信息
            yolo_lines.append(yolo_line)

    # 保存为 YOLO 格式文件
    with open(output_file, "w") as f:
        f.writelines("\n".join(yolo_lines) + "\n")

    print(f"转换完成：{json_file} -> {output_file}")


def process_folder(json_folder, image_folder, output_folder):
    """
    批量将 Labelme JSON 转换为 YOLO 格式
    :param json_folder: Labelme JSON 文件夹路径
    :param image_folder: 图像文件夹路径
    :param output_folder: YOLO 格式输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 JSON 文件夹
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        image_name = os.path.splitext(json_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_name)

        # 检查对应的图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件未找到，跳过：{image_path}")
            continue

        # 读取图像宽高
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件：{image_path}")
            continue
        image_height, image_width, _ = image.shape

        # 输出 YOLO 文件路径
        output_file = os.path.join(output_folder, os.path.splitext(json_file)[0] + ".txt")

        # 转换并保存为 YOLO 格式
        labelme_to_yolo(json_path, image_width, image_height, output_file)


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="将 Labelme JSON 转换为 YOLO 格式")
    parser.add_argument("--json_folder", required=True, help="Labelme JSON 文件夹路径")
    parser.add_argument("--image_folder", required=True, help="图像文件夹路径")
    parser.add_argument("--output_folder", required=True, help="输出 YOLO 文件夹路径")

    args = parser.parse_args()

    # 批量转换
    process_folder(args.json_folder, args.image_folder, args.output_folder)
