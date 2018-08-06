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

Iterative steps taken to build this model:

| Step | Description | Remarks |
|------|-------------|:--------|
| 1| Regression Model | Steering angle was oscillating rapidly and the car easily went off track. |
| 2| Normalization | Steering angle oscillations reduced. The car moved down the lane further than before but went off track again. |
| 3| `LeNet` model | Autonomous driving was much better than previous to trials. But the car couldn't corner curves and went into the lake every time. |
| 4| `LeNet` + Augmented data - mirror images | With augmented data, there wasn't must luck cornering the first left turn as augmenting only helped train the model for right turns. |
| 5| `LeNet` + Augmented data + Left and Right Camera Images | With Left and Right camera images and a correction factor for measurements, the car was able to corner most part of the first left turn. But at the bridge it hit the curb and stopped abruptly. |
| 6| `LeNet` + Augmented data + Left and Right Camera Images + Cropping | Driving was better than the previous step. The car crossed the bridge and drove further than the previous step. However, it still went off track at the second left turn. |
| 7| Replacing `LeNet` with Nvidia's Network Architecture | With the nvidia net, autonomous driving was vastly improved. The car was able to drive much further down the track. However, it still had trouble at a few curves. |
| 8| Using a generator with smaller batch size | With a generator, training was completed quickly. A small batch size seemed to improve the model significantly. The car stopped going off track. |
 

#### 2. Training data collection

Different approaches to get the training data:

|Approach | Description | Remarks |
|-------- |:------------|:--------|
| 1| Drove 1 lap using **arrow keys** to steer | This resulted in a model wasn't accurate and the car oscillated too much. |
| 2| Drove 1 lap using **mouse** to steer with frequent short clicks to change steering angle | The autonmous driving was better than before, but it was visible than the steering angle was changing rapidly. |
| 3| Drove 1 lap using **mouse** to steer with more constant steering angles | With this training set, the cornering of curves was smoother than before. However, at certain turns, the steering angle wouldn't receed to straighten the car. |
| 4| Drove 2 laps using **mouse**. One lap with frequent short clicks to change steering angle. One lap with more constant steering angles | This proved to be most effective to train the model. The combination of frequent steering angle changes and constant steering angles did a great job training the model! |

* Images from all three cameras were used.
* Results of various correction factors:

|Correction Factor | Remarks |
|------------------|:--------|
| 0.15 | At curves the car would hit the curb when it was at already into the curve. The model wasn't great at cornering with this low correction factor. |
| 0.20 | The car would nearly finish getting out of the curve but would hit the curb as it was ending the curve and get stuck. |
| 0.30 | The car oscillated a lot within the track. A slight steer towards either of the curbs with push the car away from it ending up with a lot oscillation in steering angle. |
| 0.28 **(optimal)** | The car was able to drive in the center of the lane and corner curves correctly with this correction factor. |

* The camera images were also augmented by flipping them. 
* Augmenting helped train the model for right turns.
* A 20% split was chosen for training and validation samples.
