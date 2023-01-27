# Applied AI in biomedicine

Final project for the *Applied AI in biomedicine* course. 

Course held @ Politecnico di Milano\
Acadamic year 2022 - 2023

## Table of contents
- [Introduction to the problem](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#introduction-to-the-problem)
- [Dependencies](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#dependencies)
- [Data](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#data)
- [Evaluation](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#evaluation)
- [Methods](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#methods)
- [Results](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#results)
- [Limitations](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#limitations)
- [Authors](https://github.com/VladMarianCimpeanu/applied_AI_in_biomedicine/blob/main/README.md#)

## Introduction to the problem
Lung diseases are often associated with excruciating pain and suffering as they affect the breathing pattern of the
patient due to suffocation and related symptoms. They are also one of the top causes of death worldwide and include
pneumonia and tuberculosis diseases.

In this project, we are required to develop a classifier able to detect and distinguish signs of *pneumonia* and *tuberculosis* from chest x-ray images.

## Dependencies
In this projet we used the following packages:
- tensorflow
- keras
- open_cv
- keras_cv
- scikit-learn
- pandas
- numpy
- PIL

**Important** consideration: we use keras_cv to perform mix_up data augmentation.\keras_cv requires **tensorflow** **v2.9+** 

## Data
The provided dataset is composed by 15470 CXR images labeled with N (*no findings*), P (*Pneumonia*) and T (*tuberculosis*) with size 400x400 distributed as follows:

![labels_distribution](https://user-images.githubusercontent.com/62434812/215167762-b759f4f9-e6b7-44c2-952f-36629ee61c65.png)


To increase the quality of the images, we use CLAHE method to increase the contrast and Gaussian blur to reduce the noise. 
## Methods
Deep-learning methods
based on convolutional neural networks (CNNs) have exhibited increasing potential and efficiency in image recognition tasks, for this reason, we implement and compare different CNN-based architectures. The notebooks where these models are trained can be found in the code folder. Finally we use grad-CAM and occlusion techniques to get explainations from our models.
![pipeline](https://user-images.githubusercontent.com/62434812/215168386-15b95452-5c0b-410f-be82-e6f5c16091be.png)


## Evaluation
Due to the high imbalance between classes, accuracy can not be considered as a good metric. More interesting are Precision, F1-score and Recall.\
Our best model reaches the following performances on the test set:

| Metrics | No findings | Pneumonia | Tuberculosis|  
|------|---------|--------|--------|
|*Precision*|0.972| 0.978| 0.943|
|*Recall*|0.980|0.985| 0.887|
|*F1-score*|0.976|0.982| 0.914|

## Results
A summary of the project's results, including any key findings or insights.
![gradcam_coronet2](https://user-images.githubusercontent.com/62434812/215168509-e5463299-96cc-40a0-9789-48f24eb34c25.png)
![gradcam_darknet1](https://user-images.githubusercontent.com/62434812/215168513-ce52bd4b-4ffa-4c60-844d-ff0cf489463d.png)
![gradcam_darknet2](https://user-images.githubusercontent.com/62434812/215168518-baf2eed1-60d6-4d62-910c-f0c2eaf67606.png)

## Limitations
We trained our models on Colab platform, providing us with nvidia tesla k80 gpu (24GB VRAM) and 12GB of RAM. Due to the size of images and the memory consumption of the models at training time, we easily run out of memory, thus, for our best models we couldn't afford a batch size greater than 32.\
This implies one epoch took us 470s on average. VRAM is not the only limitation, as matter of fact, we tried to optimize the data pipeline by caching all the images on RAM, so that the dataset iterator does not need to read images from disk, nevertheless, RAM memory was not enough, avoiding us performing this optimization.

Given this hardware limitations, we could not deeply explore the hyperparameters space and use cross validation to get more robust results.
## Authors
| Name | Surname | github | 
|------|---------|--------|
| Sofia | Martellozzo | [link](https://github.com/sofiamartellozzo)|
| Vlad Marian | Cimpeanu | [link](https://github.com/VladMarianCimpeanu)|
| Federico | Caspani | [link](https://github.com/FedericoCaspani)|
