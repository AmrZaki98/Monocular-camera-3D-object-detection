# Monocular-camera-3D-object-detection

### The yolo model is taken from the following : https://github.com/zzh8829/yolov3-tf2


### To download and Convert pre-trained Darknet weights in the above repo

##### yolov3
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf
```
## Samples of the Detection algorithim

![image](https://user-images.githubusercontent.com/90519613/194863435-a7a3905c-5d21-4b71-a134-b1abef41e2a3.png)

![image2](https://user-images.githubusercontent.com/90519613/194864665-111d9c62-6f48-47ac-8ec9-9c65997978fd.png)


## The algorithim 

1. Yolov3 algorithim is used to draw 2D bounding boxes around the object <br />
2. Another deep learning model is used to get the orientation of each detected object <br />
3. Following the geometric constraints equations from this paper: https://arxiv.org/abs/1612.00496, one could calculate the displacement between the camera and the center of the object.<br />
4. We assume that the dimention of any object is the average dimention of the object's class in the kitti datset


## references:
https://arxiv.org/abs/1612.00496 <br />
https://github.com/zzh8829/yolov3-tf2
