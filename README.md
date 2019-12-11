# Kaggle-APTOS-2019-Blindness
Top 37% entry (out of 2943) for Kaggle APTOS 2019 Blindness competition

The goal of this competition was to identify and rate the level of diabetic retinopathy from retinal images.  My best approach was an ensemble of deep learning classifiers (resnet34, efficientNetB6) trained on both original and preprocessed images.  Model ratings scored a kappa metric of 0.904 on a test set of approx. 12,000 images.  

More info can be found at the Kaggle site: https://www.kaggle.com/c/aptos2019-blindness-detection/overview

## Solution

### Data
Retinopathy causes changes in a healthy retina that can be seen in close-up retinal images.  Clinicians can assess the level of illness from characteristics of the image.

![Data](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/diabetic%20retinopathy%201.png)

Source: https://www.biorxiv.org/content/biorxiv/early/2018/06/19/225508.full.pdf

A clinician rates the presence of diabetic  retinopathy in each image on a scale of 0 to 4, according to International Clinical Diabetic Retinopathy severity scale (ICDR):
* 0 – No DR
* 1 – Mild DR
* 2 – Moderate DR
* 3 – Severe DR
* 4 – Proliferative DR
Ratings are based on human judgement of images with differing levels of brightness, orientation, and focus so there is some variation in the ratings.

Kaggle provided a data set of 3660 training images with ratings.  In addition I found a number of other data sets available online to add to the training set.  In all, a total of 44,000 image samples were obtained.  

### Data Selection
The proportion of samples with a 0 - No DR rating was significantly higher than the other ratings.  Early results showed better prediction on a validation set if the images with 0 ratings were excluded from the training set.  In order to accomplish this the Kaggle supplied 0 rated images were retained, and the other 0 rated images were excluded from further training.  In addition it was found that a number of images were duplicated and so duplicates were removed as well.  This resulted in a training set of roughly 12,000 images.

### Image Preprocessing
In order to compensage for brightness changes and to increase the contrast of the various indicators of retinopathy in the image a few preprocessing techniques were used, leveraging the opencv python library cv2.

#### Increased contrast
The first was taken from this starter example: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy.  In this example, contrast is increased by subtracting a blurred image from the original.

![Increased Contrast](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/blur%20contrast%20images.png)

Additional image processing was also used in the models:

#### Contrast Limited Adaptive Histogram Equalization (CLAHE) method -on RGB
Contrast increased using CLAHE independently on each channel of the RGB image.

python code from https://github.com/keepgallop/RIP/blob/master/RIP.py

![CLAHE](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/clahe%20processed.png)

#### CLAHE using CIELAB colorspace
In this approach the RGB image is converted to the L*A*B* color space and CLAHE is used to increase contrast on only the Lightness (L) channel.  This is motivated by the fact that CIELAB was designed so that a numerical value change corresponds to amount of perceived change. Using this approach preserves global image color while normalizing contrast on L.

![CLAHEL](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/clahel%20processed.png)

### Data Augmentation

The training images were transformed at random during training to augment the data.  Images were flipped left to right, rotated from 0 - 45 degrees, size adjusted from 100%-110%, brightness varied 100-110%.

### Convolutional Neural Networks
Two differerent architecture families were leveraged in this effort:

#### Resnet
The traditional residual NN that is pretrained on imagenet data.  https://pytorch.org/hub/pytorch_vision_resnet/
Based on preliminary training and validation, Resnet18 was satisfactory, Resnet34 provided significant improvement, and Resnet50 did not provide substantial increase in validation accuracy.  Resnet34 was used going forward.

#### Efficientnet
An improved residual NN architecture that has better scaling properties (using less compute resources for similar imagenet performance as resnet) Paper: https://arxiv.org/abs/1905.11946  Pytorch implementation: https://github.com/lukemelas/EfficientNet-PyTorch

EfficientNet-B6 was used based on success of other competitors.

#### Approach for turning an Ordinal Regression problem into a Classification problem 
Predicting Retinopathy ratings is an ordinal regression problem, since valid ratings are integers 0-4 and have an inherent order.  Convolutional neural network architectures are designed as classifiers.  if the ratings are used as classes predictions will not take advantage of the inherent order in the classes.  The model predicted probabilities for the classes might not make sense for ordinal data.  For example if rating 0 and rating 4 both have high probability of being correct while 1,2, or 3 are lower.

The apriori information that the ratings are ordinal can be encoded in a new definition of output classes.  In this case, it becomes a multi-label problem.

|Rating Input|Class Labels|Meaning|
-----------------------------------
|0| ''|r=0|
|1 |'1'|r<=1|
|2 |'1,2'|r<=2|
|3 |'1,2,3'|r<=3|
|4 |'1,2,3,4'|r<=4|


### Training and Model combinations
The overall strategy was to design an ensemble model using a variety of convolutional neural network architectures appled to differing image pre processing. The following commbinations were made, resulting in 384 models

Architectures (1) Resnet34 (2) 

### Model selection
Model selection was based on the best cross-validation scores.  The contest allowed two submissions to be considered. These were selected so one was a result of cross-validation which does not group experiments, and one where experiment grouping was allowed. 

![MPreds](https://raw.githubusercontent.com/filipmu/Kaggle-LANL-Earthquake-Prediction/master/preds.png)

Full notebook (via nbviewer): https://nbviewer.jupyter.org/github/filipmu/Kaggle-LANL-Earthquake-Prediction/blob/master/Kaggle%20LANL%20Earthquake%20Prediction%2013-best%20scoring%20submission.ipynb
