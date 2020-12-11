Introduction:
The purpose of our project was to design a computer vision model that can identify and classify multiple flower types using Mask R-CNN. 

Research, Preparation:
	We spent a few days researching and looking up different ways to implement a Mask RCNN. 
  We found a nice tutorial that we were able to use and manipulate to help us get what we needed. The tutorial was created by Tanner Gilbert. 
  See the link below for his Github. We followed his example and decided to work with Google Colab because of vast issues with Tensorflow/Keras 
  dependencies and version mismatching for current methods of using MaskRCNN. Google Colab allows for easy python package integration and we 
  could both work on the same code.   
  
https://github.com/TannerGilbert/MaskRCNN-Object-Detection-and-Segmentation/blob/master/MaskRCNN%20Using%20pretrained%20model.ipynb 

	To better understand how a Mask RCNN works we used the following article.
  
https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1 
The article does a good job of explaining what a Mask R-CNN is.



A key feature is that a Mask R-CNN doesn’t just have 2 outputs but 3. It outputs an object class, a bounding box, and a mask.
The model actually uses a CNN to predict the mask of the object. An image is put through a CNN after which outputs a feature map.
The feature map is divided up into regions of interest. Those regions of interest are fed into the CNN’s to output the mask.
The other regions of interest are warped and sent into a fully connected layer. Softmax is used to output the object class and regressor is 
used to output a single bounding box per object. 



Video Annotator
https://colab.research.google.com/drive/1dYxakDlG_N6fBCDWoNMJRnG2gNGOS0OA?usp=sharing

Sources:
Datasets:
https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html

http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

https://dataverse.harvard.edu/file.xhtml?fileId=4105627&version
=7.0

https://www.kaggle.com/alxmamaev/flowers-recognition

https://www.tensorflow.org/datasets/catalog/tf_flowers

Github Reference Code:
https://github.com/TannerGilbert/MaskRCNN-Object-Detection-and-Segmentation/blob/master/MaskRCNN%20Using%20pretrained%20model.ipynb 
https://github.com/matterport/Mask_RCNN 

Annotation tool:
https://www.makesense.ai/ 

Model Class Comparison
https://github.com/onnx/models
