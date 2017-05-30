#**Traffic Sign Recognition** 

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

[image0]: ./distributionAlongClass.png "Distribution"
[image1]: ./randomImages.png "Random"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new/aheadonly.jpg "Traffic Sign 1"
[image5]: ./new/childrencross.jpg "Traffic Sign 2"
[image6]: ./new/normalcaution.jpg "Traffic Sign 3"
[image7]: ./new/priorityroad.jpg "Traffic Sign 4"
[image8]: ./new/stop.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I check the dataset use normal python function,they include:


* The size of training set : 34799
* The size of test set : 12630
* The size of validation set : 4410
* The shape of a traffic sign image : 32x32
* The number of unique classes/labels in the data set :43
* The chanel count of image : 3
* The max value of feature : 255
* The min value of fearue : 0

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and forth code cell of the IPython notebook.  

First I use matplotlib visualize the training and validation set's distribution along class.I notice these distribution is very uneven,i think it has a impact to the trianed model,but i have not find a method to deal with it .
![alt text][image0]

Then i random select nine images from training set and show them,to get a intuition impression to the dateset
![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.
I did not grayscale the image ,because i notice the trainning result did not have much improvement

I  normalize the training, validation and test set.After this feature value locates in  a range [0,1], it will accelerate the training process.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I just use the provided training ,validation,test set my final dateset.My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

Then i augmented the trainning set ,it is in the sixth code cell of the IPython notebook. Because i want to decrease the gap between the training error and validation error ,prevent overfit . 

I use skimage.transform to add a rotated image for every image in the origin trainning set,the angle is located randomly betwwen -15 and 15，it results a doubled training set.

Here is an example of an original image and an augmented image:

![alt text][image3]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

I just select the LeCun model,after argumenting train set and tune parameters it get a 93% accuracy on validation set,so it also is my   final model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         | 32x32x3 RGB image  | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
|RELU	|  |
| Max pooling	      	| 2x2 stride,  output 14x14x6 |
| Convolution 5x5	    | 1x1 stride,valid padding,outputs 10x10x16     |
|RELU| |
|Max pooling | 2x2 stride,output 5x5x16 |
|Flatten| |
| Fully connected		| 400->120        		|
|RELU | |
|Drop out| 0.75 |
|Fully connected | 120 ->84|
|RELU||
|Drop out | 0.75 |
|Fully connected | 84--> 43|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

This step located in the eigth ,ninth,tenth ,tweleth cell

Optimizer is AdamOptimizer, batch size is 100,epochs is 30, learning rate is 0.001.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code  is located in the tenth,eleventh, twelvth cell of the Ipython notebook.

My final model results were:

* training set accuracy : 98%
* validation set accuracy : 95%
* test set accuracy : 90%

The process that i took can be summaried to three step:
1. determine batch size,epoch,learning rate to make the model get nearly 100% accuracy on train data;
2. use dropout, argumenting train data to get the model achieve more than 93% accuracy on validation data

The traffic classifer problem is like recognizing hand written digits,but its has more classes,so I decided to select LeCun network as the start model, when it is not fitable i will adjust it .
The first step is to get a nearly 100% train accuracy. I tried 10,20,30 as epoch,0.0005,0.0008,0.001 as learning rate, 50,100,200 as batch size.Then i select epoch = 30,learning rate = 0.001,batch size = 128,and got a result of 99% train accuracy and 92% validation accuracy.

I think it is time to prevent overfit ,namely to increase validation validation accuracy .First i decided to determine dropout ,and tried 0.5,0.6 and 0.75, based on accuracy and time  i select 0.75.Then i  argumented train set,i just randomly rotated every image between 0 and 360 degree.But the validation accuracy is not improved.After talking with classmates,i change the ratating scope to -15 and 15 degree,and get a 95% validation accuracy. Now i  got 98% train accuracy and 95% validation accuracy. I think it is enough.

After all the steps ,my final architecture is just LeCun Model，learning rate = 0.001，batch size = 100,epoch = 30,dropout = 0.75


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on test data is located in the 12th cell of the Ipython notebook.
But on the test set ,the accuracy is 90.5%

The code for predicting new images is located in the 13th~17th cell.Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Stop sign  		| 
| Stop  			| U-turn 			|
| Priority Road       |			Yield |
| General Caution	      		| Bumpy Road	|
| children crossing			| Slippery Road      	|


On the new images ,the model's accuracy is 0%。

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code  is located in the 17th cell of the Ipython notebook.

 The top five soft max probabilities were

| img        	|     Top Prediction	        					|Second|Third|Fourth|Fifth| 
|:----:|:-----:| ---| --- | --- |--|
| Ahead Only      		| Stop ,0.74		| No entry,0.12|End of all speed and passing limits,0.09|Right-of-way at the next intersection,0.011| Speed limit (60km/h),0.007|
| Stop  			| 		Road work,0.43	|Speed limit (80km/h),0.13|Right-of-way at the next intersection,0.12|Priority road,0.11|Dangerous curve to the right,0.046|
| Priority Road       |			Right-of-way at the next intersection,0.32|Priority road,0.14|Speed limit (60km/h),0.13|Speed limit (80km/h),0.13|Stop,0.10|
| General Caution	      		| Speed limit (80km/h),0.26	|Priority road,0.17|Dangerous curve to the right,0.08|Road work,0.09|No passing for vehicles over 3.5 metric tons,0.06
| children crossing			| Stop,0.43|      Speed limit (60km/h),0.29	|Speed limit (80km/h),0.13|No entry,0.04|End of all speed and passing limits,0.04|

The model is not sure about the result ,because the largest probability is not dominated.And when i try multly time on the new imgs ,i can get different result,it is confusing,i have not understand why can this 