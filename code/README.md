# Code
This directory is structured as follows:
- utility_package: which contains all the py modules with useful functions used across our notebooks (for the explainability and the evaluation of the models).
- cnn_and_darknet.ipynb: in this notebook we train a cnn from scratch and darknet model. 
- vgg19.ipynb: in this notebook we perform transfer learning and fine tuning on the well known vgg19 architecture.
- coronet.ipynb: in this notebook we perform transfer learning and fine tuning on a Xception-based model.
- tfr_dataset_generator.py: this python script can be used to build a TFR dataset.

## tfr dataset
We assume in this directory there is a dataset of images structured in the following way:

```
├── dataset\
│   ├── training\
│   │   ├── n\
│   │   ├── p\
│   │   ├── t\
│   ├── validation\
│   │   ├── n\
│   │   ├── p\
│   │   ├── t\
│   ├── test\
│   │   ├── n\
│   │   ├── p\
│   │   ├── t\
```

Where n, p and t are the possible labels for the task (nothing detected, pneumonia, tuberculosis).\
In this case we can use the tfr_dataset_generator.py script to generate a new dataset in the TFR format starting from the one defined above. By default, the script will resize all the images to 255x255, but it can be manually changed, by changing the parameter size in the build_tfr function. This can be useful when performing transfer learning on pretrained models which are optimized to work with images of different size such as vgg19 and Xceptio (they take 299x299 size).
