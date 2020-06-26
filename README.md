# JantaHack_ComputerVision
Emergency vs Non-Emergency Vehicle Classification

Fatalities due to traffic delays of emergency vehicles such as ambulance & fire brigade is a huge problem. In daily life, we often see that emergency vehicles face difficulty in passing through traffic. So differentiating a vehicle into an emergency and non emergency category can be an important component in traffic monitoring as well as self drive car systems as reaching on time to their destination is critical for these services.

In this problem, you will be working on classifying vehicle images as either belonging to the emergency vehicle or non-emergency vehicle category. For the same, you are provided with the train and the test dataset. Emergency vehicles usually includes police cars, ambulance and fire brigades.

Data Description

1. train.csv – [‘image_names’, ‘emergency_or_not’] contains the image name and correct class for 1646 (70%) train images. Images contains 2352 images for both train and test sets

2. test.csv: [‘image_names’] contains just the image names for the 706 (30%) test images

3. sample_submission.csv: [‘image_names’,’emergency_or_not­’] contains the exact format for a valid submission (1 - For Emergency Vehicle, 0 - For Non Emergency Vehicle)

Evaluation Metrics : Accuracy

Approach:

Since these are the baby steps towards to learn convolution neural networks, I have implemented a very basic convolution network without the use of any pretrained models. The model architecture consists of 4 layers of conv2D, BatchNormalization and MaxPooling2D stacked with flattened layer along with 2 dense layers connected to the output layer. The dense layers are sandwiched with drop out layers after each dense layer to handle the over fitting.

Further Work:

Understand and implement pretrained models like Imagenet and VGGNet etc along with using data augmentation techniques to improve the accuracy.
