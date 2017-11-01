**Behavioral Cloning Project**


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./driving_video.gif "Autonomous Driving Video"
[image3]: ./data_sample.png
[image4]: ./flipped_data_sample.png

## Rubric Points

---
**Files Submitted & Code Quality**

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

model.py implements the training of the model. I used the udacity AWS image to train the network. The paths to the training/validation data are hard coded at the top of the file.

My network uses the current speed as input, so I modified drive.py to provide it at inference time.

visualization.py takes a .h5 model file and generates a png image of the model.

images.py just flips one of the data images so it can be included in the README.

The video of the car driving autonomously is 'driving_video.mp4'. It includes one lap in each driection on the first track.

The trained model itself is in 'model.h5'.

**Model Architecture and Training Strategy**

**1. An appropriate model architecture has been employed**

I used a conventional architecture consisting of convolutions followed by fully-connected layers. The first two layers are Keras Lambda layers that perform image cropping and normalization. I crop off roughly the top third of the image. The current speed is also used as an input the the fully connected layers.

**2. Attempts to reduce overfitting in the model**

Adding dropout layers seemed to reduce performance. I had better luck with collecting varied training data and reducing the number of parameters.

**3. Model parameter tuning**

I used an adam optimizer. I did reduce the default learning rate, that seemed to work better when I was initially trying to get off the ground. The first network I had success with had about 5 million parameters. I reduced this to around 1 million by using an extra convolutional layer and reducing the number of neurons in the dense layers.

**4. Appropriate training data**

The original training data was somewhat noisy. With this this data, the car could drive around the track, but would occasionally swerve or significantly cross the yellow lines. I eventually reduced these problems by collecting cleaner data and fine tuning the original model. During data collection, I tried to use the mouse as much as possible, and also tried to have the same amount of clockwise and counter-clockwise driving data. I also included footage of the car driving on the second track.

**Model Architecture and Training Strategy**

**1. Solution Design Approach**

I first started with a network of around 5 million parameters. It had 3 convolutional layers, and 3 dense layers. With the training data I had it was pretty much already able to drive around the first track. Most of the changes I did were to try to improve driving around the second track. Some things I tried were switching to HSV instead of RGB, adding current speed as input, reducing the number of parameters and adding dropout.

I also experimented with dropping images with angles close to zero, but was able to acheive good performance without it. I also tried using the left and right camera images for recovery, but this actually worsened performance. I was using a simple hard coded adjustment to the steering angle, and most likely this parameter was not tuned correctly.

I eventually tried fine tuning my first track model with a second track only data set. This model was able to drive for long stretches on the second track, but would still go off the road in certain places. I haven't included this in the interest of simplicity.

**2. Final Model Architecture**

Here is the final model visualized with Keras. The first two layers do image cropping and normalization. This is followed by 4 convolutional layers and 3 fully connected layers. Speed is added as input to the first fully connected layer. The network has about 1 million parameters.

![alt text][image1]

**3. Creation of the Training Set & Training Process**

I drove around both tracks in both directions, using the mouse as much as possible. I also included a lot of footage of myself driving back onto the road starting from the grassy areas. The car seems to recover well if you manually throw it off course, and can sometimes find its way back onto the road, if it's in the camera frame.

I only use the center images for training, and I include a flipped version of every image. The data is loaded and augmented using a generator. The final training set consists of about 40,000 training examples (80,000 with augmentation) and 10,000 validation examples. I trained for 20 epochs on this data set. I also used two other data sets to fine tune the model, a small data set (~4000 examples) to improve behavior on certain turns, and 20,000 'high quality' examples, which only included near-perfect driving behavior. I fined-tuned for 10 epochs on both of those.

Here is an example image from the dataset, and the flipped image generated by augmentation.

![alt text][image3] ![alt text][image4]

**Final Result**

Here is a gif of the car driving in auto mode. The video cuts halfway when I put it back into manual mode to turn the car around.

![alt text][image2]

