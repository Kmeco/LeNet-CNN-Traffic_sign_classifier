# **Traffic Sign Recognition**


**Build a Neural Network for Traffic Sign Recognition**

This project has the following steps:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results


[//]: # (Image References)

[image1]: ./images/histogram_train.png "Training data histogram"
[image2]: ./images/histogram_valid.png "Validation data histogram"
[image3]: ./images/histogram_test.png "Test data histogram"
[image4]: ./images/example_raw.png "Raw RGB - as loaded"
[image5]: ./images/example_norm.png "Images after normalization and grayscaling"
[image6]: ./images/vertically_or_horizontally.png "vertically or horizontally flipable"
[image7]: ./images/vertically_and_horizontally.png "vertically and horizontally flipable"
[image8]: ./images/180.png "180 roattion invariant"
[image9]: ./images/class_change.png "class change"
[image10]: ./images/my_images.png "random images"
[image11]: ./images/my_images_proc.png "random images preprocessed"



## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) of the project individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

##### 1. Provide a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

##### 2. Include an exploratory visualization of the dataset.

Here are three bar charts showing the data distribution across the classes. We can see that the data set is very unbalanced, but the classes are proportionally distributed across the training, validating and testing subsets.

![Training data histogram][image1]
![Validation data histogram][image2]
![Testing data histogram][image3]

### Design and Test a Model Architecture

##### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

I first tested the images in full RGB with 3 channels and compared the network accuracy performance to a grayscale dataset. There was no significant drop in accuracy, but the training time decreased which is why I decided to use grayscale in the final model. All of the images were also normalized to the range (0, 1). The resulting mean was closer to zero which improves training.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![Raw_images][image4]
![Processed_images][image5]

To add more data to the data set, I used the symmetry of certain instances. There were 5 different cases:
(I found this technique in this [blog](http://navoshta.com/traffic-signs-classification/))

* vertically or horizontally symmetric
![v&h][image6]
* both vertically and horizontally
![v or h][image7]
* 180 rotation invariant
![180][image8]
* class change after flipping
![class change][image9]


##### 2. Describe what your final model architecture looks like (including model type, layers, layer sizes, connectivity, etc.)

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 image   							|
| Convolution 5x5, RELU    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5, RELU    | 1x1 stride, valid padding, outputs 10x10x16   									|
| Max pooling - flatten	| 2x2 stride,  outputs 5x5x16    => 400  									|
|				Convolution 5x5, RELU - flatten	|	1x1 stride, valid padding, outputs 1x1x400	=> 400									|
|Concatenate the last two layers				|								outputs 1x800				|
|Dropout | keep probability 0.5|
| Fully connected layer | outputs 43 logits |



##### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a trial and error approach starting with the values provided in the LeNet lesson. I was keeping a Log of all the attempts to adjust the hyperparameters more efficiently. This can be found in the Jupyter notebook after the training cell.

##### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.933
* test set accuracy of 0.925

If an iterative approach was chosen:
###### What was the first architecture that was tried and why was it chosen?
I first decided to use the LeNet architecture, because it proved to be very effective on the MINST data set. Convolutional layers are in general well suited for image recognition because of their invariance to augmentation, which is why I assumed that they would work well with traffic signs even at different angles and at different lighting conditions.
###### What were some problems with the initial architecture?
The architecture was too deep and there was a lot of overfitting because of the traffic sign images being more complex compared to the MINST.
###### How was the architecture adjusted and why was it adjusted?
In the LeCunn architecture adopted, only one fully connected layer is used instead of three. The input into the last classification fully connected layer is a concatenated vector of the ouput from the last two convolutions instead just considering the final one.
######  Which parameters were tuned? How were they adjusted and why?
The learning rate and the dropout were the two things which I focused on. A dropout probability of 0.5 has proven to be the most effective. I first used a higher learning rate to train the model faster and then decreased it for fine tuning.



### Test a Model on New Images

##### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are fifteen German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11]

I deliberately chose images which might be difficult to classify to test the strengths and weaknesses of my model. Some of the images are skewed and rotated, some have distortions such as trees in front of them and lastly there are traffic signs which are not of the German standard but would still be easily recognizable by a human.

##### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way at the next intersection 	| Beware of ice/snow| FALSE |
| Priority road | Priority road		| TRUE |
| Yield					| Yield		|TRUE |
| Stop	      		| Stop				 				|TRUE |
| Stop	| Stop     							|TRUE |
| General caution	| Children crossing |  FALSE |
| Dangerous curve to the left | General caution | FALSE |
| Double curve	| Slippery road  | FALSE |
| Road work	| Road work  | TRUE |
| Traffic signals| General caution | FALSE |
| Pedestrians	| General caution 	| FALSE |
| Speed limit (50km/h)	| Speed limit (50km/h) 	| TRUE |
| Wild animals crossing	|Wild animals crossing  | TRUE |
|Turn left ahead | Turn left ahead| TRUE|
| Keep right	| Keep right  | TRUE |


The model was able to correctly predict 9 out of 15 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set, however this was expected as the images chosen were hard to predict on purpose.

In some cases, such as the "Pedestrians crossing" instance, it seems like the model was confused by the white additional sign below the main sigh, which usually appears on the "General caution" signs. Furthermore, there were more examples of the general caution sign in the training set, which is why the model is more sensitive to it and is more likely to predict it. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The top 3 predictions for each image are shown in the last cell of the Ipython notebook.

Overall, the model is overconfident in most of the false predictions, which suggests overfitting.
