# Monocular-camera-3D-object-detection

### The yolo model is taken from the following : https://github.com/zzh8829/yolov3-tf2


### To download and Convert pre-trained Darknet weights in the above repo

##### yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights <br />
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
