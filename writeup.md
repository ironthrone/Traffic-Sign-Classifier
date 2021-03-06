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

[distribution]: ./output_images/distribution.png 
[balanced]: ./output_images/balanced_distribution.png
[random]: ./output_images/random_images.png
[image2]: ./examples/grayscale.jpg 
[augment]: ./output_images/augment_images.png 

[lossCurve]: ./output_images/LeNet_001_100_30.png

[newImages]: ./output_images/new_test_images.png
[softmax]: ./output_images/new_images_softmax.png


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

First I use matplotlib visualize the training and validation set's distribution along class.I notice these distribution is very uneven,i think it has a impact to the trianed model
![][distribution]

Then i random select nine images from training set and show them,to get a intuition impression to the dateset
![][random]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6th ,7th code cell of the IPython notebook.
I did not grayscale the image ,because i notice the trainning result did not have much improvement

I  normalize the data set to the  range [-1,1], it will accelerate the training process.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code located in the 7th ~ 9th cell. Because of the data set is very unbalanced among different classed,so  i select a balanced data set ,where every class have 250 sample
![][balanced]
In  the training process,my model only get a 80+% valid accuracy,so i decide to augment the data set. These augment data are:

*  openCV to random rotated image between -15 and 15 degree
*  skimage.exposure.rescale_intensity to add contrasted image
*  openCV to scaled image

Then i use sklean train_test_split split the data to 6:2:2 ,where training data is 6,valid and test are 2
Thus i got 27520 train image, 6880 valid iamge, 8600 test image

Here is an example of an original image and its augmented image:

![alt text][augment]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10th cell of the ipython notebook. 

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
This step located in the eigth ,12,13,14 cell

Optimizer is AdamOptimizer, batch size is 100,epochs is 30, learning rate is 0.001.I explain how i select these value in the next item
####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code  is located in the12,13,14 cell of the Ipython notebook.

My final model results were:

* training set accuracy : 99%
* validation set accuracy : 98%
* test set accuracy : 97%

It has been improved a lot compare to my  first sumbit:

* training set accuracy : 98%
* validation set accuracy : 95%
* test set accuracy : 90%


This is what i do before my first submit ,the process that i took can be summaried to two step:
1. determine batch size,epoch,learning rate to make the model get nearly 100% accuracy on train data;
2. use dropout, argumenting train data to get the model achieve more than 93% accuracy on validation data

The traffic classifer problem is like recognizing hand written digits,but its has more classes,so I decided to select LeCun network as the start model, when it is not fitable i will adjust it .
The first step is to get a nearly 100% train accuracy. I tried 10,20,30 as epoch,0.0005,0.0008,0.001 as learning rate, 50,100,200 as batch size.Then i select epoch = 30,learning rate = 0.001,batch size = 128,and got a result of 99% train accuracy and 92% validation accuracy.

I think it is time to prevent overfit ,namely to increase validation accuracy .First i decided to determine dropout ,and tried 0.5,0.6 and 0.75, based on accuracy and time  i select 0.75.Then i  argumented train set,i just randomly rotated every image between 0 and 360 degree.But the validation accuracy is not improved.After talking with classmates,i change the ratating scope to -15 and 15 degree,and get a 95% validation accuracy. Now i  got 98% train accuracy and 95% validation accuracy. I think it is enough.

Then after first submit  failed ,i selected a balanced train set, like above mentioned. The time i did not tune the parameter, just augment data.
Like my first submit data, at first i only add rotate image to  the new balanced data set only have a 83% validation accuracy. Then i add contrast image using skimage.exposure.rescale_intensity(), and resized image ,finally my model get a 99% training accuracy,98% valid accuracy,98% test accuracy

![][lossCurve]


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Code is located in 15,16,17 cell
Here are five German traffic signs that I found from google streetview :
![][newImages]

These feature may be a challenge to my model

* For image 1,3,4,5,they are all inclined
* traffic sign in image 2 not located in the middle of the image
* Image 3 is obscure


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on test data is located in the 12th cell of the Ipython notebook. My test set get a 98% accuracy

After predicting test set ,i test the five new images.the code is located in the 18th cell.
Here are the results of the prediction:

| Image			        |     Prediction	      	|   Right?  |
|:---------------------:|:------------------| -----:| 
| Priority Road       |  Priority Road | Right
|Stop  			| Priority Road| Wrong
| No Entry	      		| Dangerous curve to the left	| Wrong
| Ahead Only      		| Ahead Only   		| Right
| Speed limit (30km/h)		|      Dangerous curve to the left 	| Wrong

The accuracy is 40%,compare to the test set 96%,it is very low.But the test image count is two little, it is accident factor


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code  is located in the 17th cell of the Ipython notebook.

To make it easy to understand ,i make bar charts for each image's sofmax:
![][softmax]
We can see, the model is very confident on predicting image 1 and image 4,the largest softmax is nearly 1.0; 
It is also very certain about image 2 ,but it is a wrong predict; 
It is some kind of confidence  about the image 3 and image 5,all the top 5 softmax are high,but these result are wrong
