# multi_npu_demo
FuriosaAI Warboy와 SDK를 사용한 Object Detection 데모 코드가 담긴 레포지토리입니다.
자세한 코드에 대한 설명은 [슬라이드]()를 통해 확인 가능합니다.

## 0. 의존성 설치하기
```shell
pip install -r requirements.txt
```

# YOLOv7

## 1. 모델 준비 하기
Object Detection 모델로는 공개되어 있는 [YOLOv7](https://github.com/WongKinYiu/yolov7)을 사용하고 있습니다.
1. YOLOv7 Weight 파일 다운로드 및 ONNX Export
```
$ cd model_export
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
$ python onnx_export.py --weights=yolov7.pt --onnx_path=yolov7.onnx
```
2. Calibration 데이터 다운로드<br/>
: Calibration을 위해 coco val2017 데이터 셋을 다운로드합니다.
```
$ cd ..
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
```

3. YOLOv7 모델 Quantization 및 컴파일<br/>
  : coco val 2017 데이터셋을 사용하여 Calibration과 Quantization을 진행합니다. 이후, ```furiosa-compiler```를 사용하여 ```.enf``` 파일을 생성합니다.
```
$ cd model_export
$ python furiosa_quantizer.py --onnx_path=yolov7.onnx --output_path=yolov7_i8.onnx --calib_data=../val2017
$ furiosa-compiler yolov7_i8.onnx -o ../demo/yolov7.enf --target-npu=warboy
```

## 2. Object Detection 데모 실행
```
$ cd demo
$ python run_demo.py
```

# YOLOv8

## 1. 모델 준비 하기
Object Detection 모델로는 공개되어 있는 [YOLOv8n](https://github.com/ultralytics/ultralytics)을 사용하고 있습니다.
1. YOLOv8 Weight 파일 다운로드 및 ONNX Export
```
$ cd model_export_yolov8
$ wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
$ python onnx_export.py --weights=yolov8n.pt --onnx_path=yolov8n.onnx
```
2. Calibration 데이터 다운로드<br/>
: Calibration을 위해 coco val2017 데이터 셋을 다운로드합니다.
```
$ cd ..
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
```

3. YOLOv7 모델 Quantization 및 컴파일<br/>
  : coco val 2017 데이터셋을 사용하여 Calibration과 Quantization을 진행합니다. 이후, ```furiosa-compiler```를 사용하여 ```.enf``` 파일을 생성합니다.
```
$ cd model_export_yolov8
$ python furiosa_quantizer.py --onnx_path=yolov8n.onnx --output_path=yolov8n_i8.onnx --calib_data=../val2017
$ furiosa-compiler yolov8n_i8.onnx -o ../demo/yolov8n.enf --target-npu=warboy
```

## 2. Object Detection 데모 실행
```
$ cd demo_yolov8
$ python run_demo.py
```
