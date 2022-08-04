# Aerial Waste Model

The classifier exploits ResNet50 as the network backbone and augments it with a Feature Pyramid Network (FPN) architecture. FPN improves performances in object detection when  different scales must be taken into account and thus can benefit also classification tasks in which objects of the same class appear with variable sizes.
This repository contains a ResNet50+FPN architecture.
![fpn (2)](https://user-images.githubusercontent.com/62382874/135090779-132cbc65-4ea1-4b2b-bad2-f8bc19bd00b8.png)

The architecture was trainned with a private dataset of remote sensing scene classifition for illegal landfills detection.

**Dataset:** The dataset used to train the model is AerialWaste [https://aerialwaste.org/]
- Data augmentation: Flip and rotation and random crop (600, 800, 1000).

**Training details:** The model was trained using two GPUs Nvidia GeForce RTX 2080Ti and the following parameters:
- Learning rate: 0.005
- Batch size of 12 (given the capacity of our server)
- Loss function: Binary Cross Entropy 
- Early stopping patience: 10  
- Early stopping min delta: 0.0005
- Pretrained model:  ImageNet pre-trained weights for ResNet50 backbone. Freezing the first two layers -the code of the backbone and the pre-trained weigths can also be found in this repository-. 

A sigmoid function is applied over the last FC layer of the CNN to obtain the actual prediction of the model, that for the binary case of illegal landfills is a value between 0 and 1. 

**Execution:** The notebook "execute_model.ipynb" provides an example of how to load the trainned model and execute it on an image to obtain the classification score as well as the CAMs.
We provided only two example images using Google Maps instead aof the source used for training, given that the dataset contains sensitive data provided to us under a non-disclousure agreement.

## License
Creative Commons CC BY licensing scheme (see LICENSE). 

## Cite us
```
@misc{torres2022dataset,
  title={AerialWaste: A dataset for illegal landfill discovery in aerial images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  year={2022},
}
```
Visit our site for more details: https://aerialwaste.org/
