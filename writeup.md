# **Traffic Sign Recognition**

## Write-up Template

### You can use this file as a template for your write-up if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/histogram.png "Visualization"
[image2]: ./images/sample-unprocessed.png "Original Sample"
[image3]: ./images/sample-preprocessed.png "Pre-processed"
[image4]: ./images/sign0-formatted.png "Traffic Sign 0"
[image5]: ./images/sign1-formatted.png "Traffic Sign 1"
[image6]: ./images/sign2-formatted.png "Traffic Sign 2"
[image7]: ./images/sign3-formatted.png "Traffic Sign 3"
[image8]: ./images/sign4-formatted.png "Traffic Sign 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Jazzbert/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34,799** images
* The size of the validation set is **4,410** images (~12.7% of training)
* The size of test set is **12,630** images
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes in the data set is **43**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the different classes are not represented evenly.  Several classes have very low representation in the training population.  For example, while most speed limit signs are well represented, 20kph sign (index 0) has only a very small set of training examples.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The execution for preprocessing for this step is contained in the ninth code cell of the IPython notebook, with some functions helper functions and the preprocessing pipeline in the seventh and eight code cells.

I tried using several different techniques for adding additional training images, and each time, I actually got worse performance.  I expect there are additional steps I can take to try some different approaches, including specifically adding modified images from classes that are poorly represented in the training set.

In any case, per minimum requirements, I scaled the data over -1 to 1.  This did have dramatic increase in training performance because in narrowed the scale of the training algorithm.  

One step I tried as well, which I'm not convinced is having a positive effect, but at least isn't making things worse at this point is implementing PCA with whitening.  In concept it seems like I should be getting better results with more evenly balanced data around zero-center.

Here are image outputs, before and after, from pre-processing:

![alt text][image2] ![alt text][image3]

## TODO: Update images for unprocessed and processed from final run

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The dataset provided for this lab was already split into training, validation, and test sets.  Since there was no specific requirement in the rubric, no steps were taken to randomize data across those data sets.

The code to load the different data sets is in the first section of the notebook, in the second code cell. The sizes of the different sets are noted above.  

Pre-processing was executed on each dataset to ensure consistent results.  The training set was shuffled each time to make sure there wasn't an issue with over-fitting represented by a consistent ordered set of training data.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eleventh cell of the Ipython notebook.  
My final model consisted of the following layers:

## TODO: Update final architecture when complete

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, 12 depth, outputs 28x28x12 	|
| Dropout					|	Keep probability 90%											|
| RELU	      	|    				|
| Max Pooling	    | 2x2 size, 2x2 stride, outputs 14x14x12      									|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x32						|
| RELU				|         									|
|	Dropout					|	Keep probability 90%	|
|	Max Pooling				| 2x2 size, 2x2 stride, outputs 5x5x32	|
| Flatten   | Reshape to single 800 length array |
| Fully Connected | Starting weights randomized, starting biases zero, outputs 400 |
|	Dropout					|	Keep probability 90%	|
| RELU    |       |
| Fully Connected | Starting weights randomized, starting biases zero, outputs 120 |
|	Dropout					|	Keep probability 90%	|
| RELU    |       |
| Fully Connected | Starting weights randomized, starting biases zero, outputs 84 |
|	Dropout					|	Keep probability 90%	|
| RELU    |       |
| Fully Connected | Starting weights randomized, starting biases zero |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training this model is in the fifteenth code cell.  The pipeline for training is setup in the thirteenth code cell and the evaluation model in the fourteenth code cell.

The various hyperparameters are contained in the tenth code cell.  While the validation accuracy continued to increase over with an ever increasing number of epochs, I decided to keep the number of epochs around 50 to limit the tendency to over-fit.

Batch size, dropout probability, and learning rate were determined through experimentation.  While not in the scope of this project, it seems like a next step to implement optimization code to narrow in on optimal values after changes in the model.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixteenth cell of the Ipython notebook.

## TODO: Update final accuracy numbers and description of approach when complete

My final model results were:
* training set accuracy of **99.4%**
* validation set accuracy of **93.3%**
* test set accuracy of **92.5%**

I started with the LeNet model and modified some steps.  The LeNet model seemed like a good starting point as it is a leading deep learning model when dealing with image recognition, plus it was strongly recommended! ;)

I had significant difficulty achieving required validation accuracy of 93% without what I thought would be causing over-fitting.  I had added several layers of dropout and eventually doubled the initial layers in convolutions plus adding two more fully connected layers.  As noted below, I still think I have a problem with over-fitting, and given more time would focus in that area.

As part of this project tried different activation functions.  Replacing RELU with softplus, RELU6 and other functions.  Most times RELU was best, but in the end softsign was best performing for the current model.

I also tried switching between AdamOptimizer and GradientDecentOptimizer.  For a while I was getting better performance with Gradient Decent, but ultimately went back to Adam.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

I think the first image of my sample ("Road Work") is one of the hardest to classify, because the sign shape is typical of many different sign types, but the image within the sign seems like it could be hard to distinguish in a low-res image.

The second ("80 km/h") and forth ("30 km/h") image I would expect should be relatively easy to classify.  The second image is the cleanest image, though as it seems to be computer generated.  The errors may come from confusing similar numbers.  30 and 80 can be confused as well as 60 with both.  

The third image ("Right of Way at Next Intersection") seems to be pretty well defined and I would think should be fairly easy to classify.

The last image("No entry") should be the easiest, as it is highly distinctive.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eighteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road Work     		| Vehicles > 3.5 metric tons prohibited   									|
| 80 km/h     			| **80 km/h** 										|
| Right of Way at Next Intersection		| Vehicles > 3.5 metric tons prohibited					|
| 30 km/h	      		| 60 km/h					 				|
| No Entry			| 60 km/h      							|

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of **20%**. This does not compare favorably to the accuracy on the test set.  :(

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the twentieth cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

## TODO: put in the appropriate tables here after final run

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
