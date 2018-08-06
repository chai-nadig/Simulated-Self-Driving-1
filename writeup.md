# **Simulated Self Driving 1** 

---


The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---

#### 1. Files Included

These files are required to run this project


| Filename | Description  |
|----------|:-------------|
| `model.py` | script to create and train the model |
| `drive.py` | script to drive the car in autonomous mode |
| `model.h5` | trained convoluation neural network model file |

Input and output files

| Filename | Description |
|----------|:------------|
| `train4.zip` | The zip file of images which are used to train and obtain `model.h5` <br/> **Note:** This file isn't checked into the repo. Download it [here](https://drive.google.com/file/d/1zg2ouWtWFiOho7PPJa0Uqxp6Iqz-7UAY/view?usp=sharing).  |
| `video.mp4` | The video file of the car driving in autonomous mode |

#### 2. Training
* Download this simulator provided by Udacity - [udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim).
* Extract `train4.zip` into the project folder.
* The final directory structure should look like this 
```
Simulated-Self-Driving-1
|--- model.py
|--- drive.py
|--- model.h5
|--- train4
     |--- driving_log.csv
     |--- IMG
```
* Run the following command in the terminal
```bash
python model.py
```
* This will produce a new `model.h5`

#### 3. Driving in autonomous mode
* Run the simulator and select "Autonomous Mode".
* Run the follow command in a separate terminal.
```sh
python drive.py model.h5
```

### Architecture

#### 1. Neural Network

* The neural network used to train the model was developed at Nvidia.
* It's described in detail [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
* It contains the following layers

| Layer | Description |
|-------|-------------|
| Normalization | Keras Lambda layer  |
| Cropping | Crop the input images to discard pixels that don't contribute to the training |
| 2D Convolution | `5x5` filter with `relu` activation |
| 2D Convolution | `5x5` filter with `relu` activation |
| 2D Convolution | `5x5` filter with `relu` activation |
| 2D Convolution | `3x3` filter with `relu` activation |
| 2D Convolution | `3x3` filter with `relu` activation |
| Flatten | &nbsp; |
| Fully Connected Layer | `1164` outputs |
| Fully Connected Layer | `100` outputs |
| Fully Connected Layer | `50` outputs |
| Fully Connected Layer | `10` outputs |
| Fully Connected Final Output Layer | `1` output |
 
* In `model.py` lines `85-104` construct the neural network.
* The network includes RELU activation layers in each of the 2D Convolution layers to introduce nonlinearity.

#### 2. Reducing overfitting

* To reduce overfitting, images from all three cameras are used to train the model.
* There is a correction factor of `0.28` added and subtracted from the left and right camera images.
* Further more, the dataset is augmented with the mirror images of the three camera images.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line `106`).

### Modelling and Training Strategy

#### 1. Model Design Approach

Here're the iterative steps taken to build this model

| Step | Description | Remarks |
|------|-------------|:--------|
|Step 1| Regression Model | Steering angle was oscillating rapidly and the car easily went off track |
|Step 2| Normalization | Steering angle oscillations reduced. The car moved down the lane further than before but went off track again |
|Step 3| `LeNet` model | Autonomous driving was much better than previous to trials. But the car couldn't corner curves and went into the lake every time. |
|Step 4| `LeNet` + Augmented data - mirror images | With augmented data, there wasn't must luck cornering the first left turn as augmenting only helped train the model for right turns |
|Step 5| `LeNet` + Augmented data + Left and Right Camera Images | With Left and Right camera images and a correction factor for measurements, the car was able to corner most part of the first left turn. But at the bridge it hit the curb and stopped abruptly. |
|Step 6| `LeNet` + Augmented data + Left and Right Camera Images + Cropping | Driving was better than the previous step. The car crossed the bridge and drove further than the previous step. However, it still went off track at the second left turn. |
|Step 7| Replacing `LeNet` with Nvidia's Network Architecture | With the nvidia net, autonomous driving was vastly improved. The car was able to drive much further down the track. However, it still had trouble at a few curves |
|Step 8| Using a generator with smaller batch size | With a generator, training was completed quickly. A small batch size seemed to improve the model significantly. The car stopped going off track. |
 

#### 2. Training data collection

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
