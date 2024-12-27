## What's New

**Environment Requirements: python >= 3.6  pytorch >= 1.7**

#### **PTH Inference:**

Run `SKY_BVP_pth.py` or use the following command:

python SKY_BVP_pth.py --detect_model path/to/your/.pth --image_path path/to/your/images --output results --img_size 640


#### **ONNX Inference:**

Run `SKY_BVP_onnx.py` or use the following command:

python SKY_BVP_onnx.py --detect_model path/to/your/.pth --image_path path/to/your/images --output results --img_size 640


#### **Annotation Script:**

Run `SKY_LABEL_TOOLS.py` or use the following command:

python SKY_LABEL_TOOLS.py --detect_model path/to/your/.pth --image_path path/to/your/images --output results --img_size 640


`yolo2json.py` imports pre-annotation results into LabelMe.

`json2yolo.py` converts LabelMe annotations to YOLO format for training.

#### **Detection and Recognition:**

Run `SKY_BVP_REC.py` or use the following command:

python SKY_BVP_REC.py --image_folder <path_to_input_folder>
--output_folder <path_to_output_folder>
--weights <path_to_detection_weights>
--rec_weights <path_to_recognition_weights>
--img_size 640


#### **Training Script:**

Data path: `data/corners.yaml`  
Place images and YOLO annotation files in the same folder and modify the paths in the YAML file.  
Model definition: `models/yolov5s.yaml`  
Parameter tuning: `data/hyp.scratch.yaml`

Run `train.py` or use the following command:

python train.py


#### **Export Script:**

Run `export.py` or use the following command:

python export.py


## **License Plate Recognition Training**

Link to License Plate Recognition training:

[USPlateRec](https://github.com/Liiiiiiiu/USPlateRec.git)

## Deployment

**TensorRT** deployment can be found at [Tear_tensorrt_inference](https://github.com/Liiiiiiiu/Tear_tensorrt_inference.git)
