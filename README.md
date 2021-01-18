# Implementaion of YoloV4 on Crowd Human Dataset

## About Darknet :

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png) 

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

## Crowd Human Dataset : 

From the CrowdHuman [website](http://www.crowdhuman.org/):

CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

## Train/Validation :

The CrowdHuman dataset can be downloaded from the [here](http://www.crowdhuman.org/download.html).

The training set is divided into 3 files, and are between 2-3GB zipped.
  
  * `CrowdHuman_train01.zip`
  * `CrowdHuman_train01.zip`
  * `CrowdHuman_train01.zip`
 
A validation set is also provided in `CrowdHuman_val.zip`

Both the **training** and **validation** sets come with annotations.
  
  * `annotation_train.odgt`
  * `annotation_test.odgt`

## How to train YOLOv4 for custom objects detection in Google Colab

1. Setting up YoloV4 files:

  * Download the YoloV4 and prepare the contents
  * Upload it to Google Drive
  * Open darknet-master folder which we have just downloaded and from that open cfg folder now in the cfg folder make a copy of the file yolo4-custom.cfg now rename the copy file to yolo-obj.cfg
  
      * Open the file yolo-obj.cfg and change max_batches to (classes*2000),if you have 6 object classes change max_batches=12000.
      * Then change the line steps to (0.8*max_batches ,0.9*max_batches) ie; if you have 6 classes steps=9600,10800.
      * Set network size width=416 height=416.
      * Change line classes=80 to your number of objects in each of 3 yolo layers.
      * Change [filters=255] to filters=(classes + 5)x3 in the 3 convolutional layer immediately before each 3 yolo layers.If you have 6 classes filters=33.
      
  * Download the pre trained weights from the link [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) and save it in the darknet-master 
  folder
  * Open wordpad and type the name of each object in separate lines and save the file as obj.names in darknet-master->data folder
  * Create file obj.data in the folder darknet-master->data, containing the given text (replace classes = number of objects)
  
    ``` 
    classes= 6
    train  = data/train.txt
    names = data/obj.names
    backup = backup/ 
  
  * Create a folder in the directory darknet-master->data named obj now inside this obj folder put all your images and the respective txt files you got from labeling in step1
  * Now we have to create a train.txt file.This file directs to all training images as shown in the below picture.The easiest way to achieve this is store all images in a folder in computer open command prompt navigate to the folder using ‘cd’ and type command ‘ls’ if in linux and ‘dir’ if in windows.This will display all image names copy that and paste in text file and add ‘data/obj/’ to each line for this Find and replace option could be used.The train.txt file is stored in the darknet-master->data folder.
    
    ```
    data/obj/img1.jpg
    data/obj/img2.jpg
    data/obj/img3.jpg
    data/obj/img4.jpg
    ..
    ..
    
  * In the darknet-master folder open Makefile in wordpad and change GPU=1,CUDNN=1,OPENCV=1 as shown in the following picture.This is done to make the training on GPU.
  
    ```
    GPU=1
    CUDNN=1
    CUDNN_HALF=0
    OPENCV=1
    AVX=0
    OPENMP=0
    LIBSO=0
    ZED_CAMERA=0 # ZED SDK 3.0 and above
    ZED_CAMERA_v2_8=0 # ZED SDK 2.X
    
 2. Setting up google colab :
 
    * Upload to google drive the darknet-master.To upload click the + icon on top left and choose upload folder option.
    * Open a colab notebook by clicking on the [link](https://colab.research.google.com/notebooks/intro.ipynb).Now on the file option select new notebook.If you are already signed it will open the new jupyter notebook.
      * In the edit menu at top left select notebook settings and turn the hardware accelerator to GPU.This will help us to utilize a GPU in colab
      
      ![GPU](https://miro.medium.com/max/469/1*OEZfyV5zt3yiN7UKQwxFDQ.png)]
               
    * Now we have to mount our drive to colab.Use the following code for that.This will generate a link open that and copy the confirmation code back to the cell.
    
        ```
        from google.colab import drive
        drive.mount('/content/drive')
    
    * Now we have to change the directory to darknet-master.For that run the following code in a new cell.If your path to darknet-master is different than the given.Just find the folder in the left side tab and right click and select option copy path and change code accordingly.
    
        `%cd "/content/drive/My Drive/darknet-master/"'
    
   * Compile darknet by using following code
      
        ```
        !make
        !chmod +x ./darknet
   
   * Now we have to convert certain files to unix for that execute the following codes
   
      ```
      !sudo apt install dos2unix
      !dos2unix ./data/train.txt
      !dos2unix ./data/obj.data
      !dos2unix ./data/obj.names
      !dos2unix ./cfg/yolo-obj.cfg    
      
## Training :

The training process could take several hours even days.But colab only allow a maximum of 12 hours of running time in ordinary accounts.What we could do is training by parts.after each 1000 epoch weights are saved in the backup folder so we could just retrain from there.For starting the training run the code,

`! ./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137 -dont_show`

## Testing :

For testing run the following code(replace ‘12000’ with latest on the backup folder)and enter the path of images we want to test the results will be shown right away and will be stored as a image named predictions.jpg

`!./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_12000.weights`

## Results :

![predictions](https://user-images.githubusercontent.com/62458537/104943498-dd461680-59db-11eb-89a8-fd0cc27cc0ba.jpg)


## References :
  
  * [YoloV4](https://arxiv.org/abs/2004.10934)
  * [How to train YOLOv4 for custom objects detection in Google Colab]()
  * [Using YOLOv3 for real-time detection of PPE and Fire](https://towardsdatascience.com/using-yolov3-for-real-time-detection-of-ppe-and-fire-1c671fcc0f0e)
  * [Object detection with YOLO](https://medium.com/analytics-vidhya/object-detection-with-yolo-aa2dfab21d56)
  * [PP-YOLO Surpasses YOLOv4 — Object Detection Advances](https://towardsdatascience.com/pp-yolo-surpasses-yolov4-object-detection-advances-1efc2692aa62)
 
