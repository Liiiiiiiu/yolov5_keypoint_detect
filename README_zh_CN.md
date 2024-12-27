## What's New

**环境要求: python >=3.6  pytorch >=1.7**

#### **pth推理:**

运行SKY_BVP_pth.py 或者运行如下命令行：

```
python SKY_BVP_pth.py --detect_model path/to/your/.pth  --image_path path/to/your/images --output results --img_size 640
```

#### **onnx推理:**

运行SKY_BVP_onnx.py 或者运行如下命令行：

```
python SKY_BVP_onnx.py --detect_model path/to/your/.pth  --image_path path/to/your/images --output results --img_size 640
```

#### **标注脚本:**

运行SKY_LABEL_TOOLS.py 或者运行如下命令行：

```
python SKY_LABEL_TOOLS.py --detect_model path/to/your/.pth  --image_path path/to/your/images --output results --img_size 640
```

yolo2json.py 将预标注结果导入labelme;

json2yolo.py 从labelme转为yolo格式训练;


#### **检测与识别:**

运行SKY_BVP_REC.py 或者运行如下命令行：

```
python SKY_BVP_REC.py --image_folder <path_to_input_folder> \
               --output_folder <path_to_output_folder> \
               --weights <path_to_detection_weights> \
               --rec_weights <path_to_recognition_weights> \
               --img_size 640

```

#### **训练脚本:**

数据路径: data/corners.yaml 
    将图片和yolo标注文件放在同一个文件夹下,修改yaml文件中的路径;
模型定义: models/yolov5s.yaml
参数调节: data/hyp.scratch.yaml

运行train.py 或者运行如下命令行：

```
python train.py 
```

#### **导出脚本:**

运行export.py 或者运行如下命令行：

```
python export.py 
```

## **车牌识别训练**

车牌识别训练链接如下：

[USPlateRec](https://github.com/Liiiiiiiu/USPlateRec.git)

## 部署

**tensorrt** 部署见[Tear_tensorrt_inference](https://github.com/Liiiiiiiu/Tear_tensorrt_inference.git)





