# 使用说明_chen

## 思路说明
`align`下`detect_face`是MTCNN的Tensorflow实现  
[MTCNN_Github](https://github.com/kpzhang93/MTCNN_face_detection_alignment)  
`LFW`数据集的默认处理**疑似**仅仅是更改后缀为`.jpg`,数据集路径更改为`lfw_mtcnnpy_160`  
服务器可运行命令:
```
# 1. export PYTHONPATH=/home/chen/smartcity/facenet/smartcity
# 2. test

python3 src/validate_on_lfw.py \
../datasets/lfw/lfw_mtcnnpy_160 \
./models/facenet/20180402-114759 \
--distance_metric 1 \
--use_flipped_images \
--subtract_mean \
--use_fixed_image_standardization
```
## 代码使用说明

### 训练自己的图片
训练：
````
python src/classifier.py TRAIN <数据集> <预模型:20180402-114759.pb> <保存模型:face_classifier.pkl> --batch_size 1000 
````

### 测试自己的图片
````
python src/classifier.py CLASSIFY <数据集> <预模型:20180402-114759.pb> <保存模型:face_classifier.pkl> --batch_size 1000
````

### LFW 数据集测试
````
python src/validate_on_lfw.py <数据集> <模型:20180402-114759>  --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization
````

### facenet
`get_dataset`   
获取数据集，会自动替换参数中`~ ~user`为路径  
返回 ·list(ImageClass(class_name, image_paths))·

### detect_face
`detect_face` 检测图像中的面部，并返回边界框和点。  
## 问题
### 使用问题
- `align_dataset_mtcnn` 对其人脸 `gallery` 少了96(245-378)个
- `align_dataset_mtcnn` 对其人脸 `probe` 少了206(2642-2438)个
### Q1 import cv2
缺少软件包安装如下：  
`apt-get install libsm6`  
`apt-get install libxrender1`  
`apt-get install libxext-dev`  
[参考链接](https://blog.csdn.net/yuanlulu/article/details/79017116)  
