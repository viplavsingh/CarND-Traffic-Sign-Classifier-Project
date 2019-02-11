# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/trainingSamplesPerClass.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/augmented.jpg "Augmentation"
[image4]: ./Traffic_images/Lenet_1.jpg "Traffic Sign 1"
[image5]: ./Traffic_images/Lenet_2.jpg "Traffic Sign 2"
[image6]: ./Traffic_images/Lenet_3.jpg "Traffic Sign 3"
[image7]: ./Traffic_images/Lenet_4.jpg "Traffic Sign 4"
[image8]: ./Traffic_images/Lenet_5.jpg "Traffic Sign 5"


---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset summary

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training dataset is represented as per the class.
There are 43 classes. Each classes have some training data asscociated with it. The following bar chart shows the number of training
examples per class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1.Preprocessing


Normalization:
I have normalized the training set by by dividing the pixels by 255. i.e.(pixel/255). 

RGB to grayscale:
The RGB image has been converted to grayscale by using tf.image.grayscale() function at the time of training.

![alt text][image2]

Data Augmentation:
I have added the additional data into the training set as some of the classes have very few number of samples as compared to others.

For this purpose I have augmented the training set by rotating by a fine angle (not at the right angles) and by using the perspective transform on that rotated image. With augmentation, the training dataset becomes balanced with each classes have 5000 training examples
associated with them.

Here is an example of the augmented image:

![alt text][image3]



#### 2. final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:
 
Input is 32x32x3 RGB image.
32x32x3 RGB image is converted to grayscale by tf.image.rgb_to_grayscale.
Convolution layer 1: filter of (5x5) size with a stride of (1x1) and same adding
output of layer 1: (32x32x32)
Activation layer: Relu
Maxpool: stride of (2x2) with same padding outputs (16x16x32)
Convolution layer 2: filter of (5x5) size with a stride of (1x1) and same adding
output of layer 2: (16x16x64)
Activation layer: Relu
Maxpool: stride of (2x2) with same padding outputs (8x8x64)
Convolution layer 3: filter of (5x5) size with a stride of (1x1) and same adding
output of layer 3: (8x8x128)
Activation layer: Relu
Maxpool: stride of (2x2) with same padding outputs (4x4x128)
fully connected layer1: outputs with 1024 units
Activation: RELU
Dropout: 0.5
fully connected layer2: outputs with 512 units
Activation: RELU
Dropout: 0.5
fully connected layer 3: outputs with dimension 43 classes.


#### 3. Train the Model.

To train the model, I have used the following hyperparameters:
EPOCHS=25
BATCH_SIZE=128

Training is done in the batches of 128 here. Sample of 128 from the training set is fed into the model at a time. As a result of forward propagation, we get the logits. With the logits, we can get the cross-entropy. Our goal is to minimize this cross-entropy by making changes in the weights at each layer using back propagation. This process is repeated for all the batches across the mentioned epochs.
tf.nn.softmax_cross_entropy_with_logits is used for calculating the cross entropy.
tf.train.AdamOptimizer(learning_rate = rate) is used as the optimizer which takes learning rate as a parameter.
Finally optimizer.minimize() is called to minimize the crosss-entropy. 

#### 4. Approach taken for finding a solution

My final model results were:
* training set accuracy of 98.6
* validation set accuracy of 96.5
* test set accuracy of 95.2


Initially I chose the Lenet-5 architecture. I used the filter size of (5x5) with two convolution layer and two fully connected layer.
The validation accuracy in this scenarion was 89%. Then I used the normalization (dividing the pixel by 255) and converted the RGB image
to grayscale image. After this preprocessing I got the validation accuracy of 91%.

In the next step, I used the data augmentation to increase the dataset and to balance the dataset per class. I performed the augmentation
by rotating the image by a fine angle (not at the right angles) and then perform the perspective transform on that rotated image.
By doing this, the training dataset was balanced and increased samples. I used the same Lenet-5 based model on this new dataset.
This time, training accuracy became 99% and validation accuracy was still near about 90%. Then I used dropout with keep probability of 0.8
to reduce the overfitting. By doing this, training accuracy became little less but validation accuracy was still in the range of 90%-91%.

I changed the architecture with the addition of new convolution layer and updated number of output units. With this new model, I got 96.5% validation accuracy. A dropout in the fully connected layers with a keep probability of 0.5 was used. Learning rate of 0.001 and 15 epoch was used. The addition of additional convolutional layer and the increase of the feature maps leads to the increase of validation accuracy.

In the initial model, output units at the various layers were: (6,16,120,84).
after the modificaiton, output units at the various layers are: (32,64,128,1024,512).
 

### Test a Model on New Images

#### 1.  German traffic signs.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Model's prediction.

Here are the results of the prediction:

| Image                                 |     Prediction                                | 
|:--------------------------------------|:---------------------------------------------:| 
| Speed limit (70km/h) 	                | Speed limit (70km/h)                          | 
| Road work                             | Road work                                     |
| Right-of-way at the next intersection | Right-of-way at the next intersection 		|
| Stop                                  | Vehicles over 3.5 metric tons prohibited      |
| Yield                                 | Yield                                         |



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Image related to Stop traffic signal was not predicted correctly. It gave a wrong label of 'Vehicles over 3.5 metric tons prohibited'.

#### 3. softmax probabilities for each prediction.


For the first image, the model is relatively sure that this is a Speed limit (70km/h) sign (probability of 0.99).
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (70km/h) 			                | 
| .01     				| Road work										|
| 0 					| Right-of-way at the next intersection         |
| 0 	      			| Vehicles over 3.5 metric tons prohibited		|
| 0 			        | Yield              							|

