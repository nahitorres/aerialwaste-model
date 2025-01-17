# Aerial Waste Model

The classifier exploits ResNet50 as the network backbone and augments it with a Feature Pyramid Network (FPN) architecture. FPN improves performances in object detection when  different scales must be taken into account and thus can benefit also classification tasks in which objects of the same class appear with variable sizes.
This repository contains a ResNet50+FPN architecture.

![fpn (2)](https://user-images.githubusercontent.com/62382874/135090779-132cbc65-4ea1-4b2b-bad2-f8bc19bd00b8.png)

**Dataset:** The architecture was trained with AerialWaste [https://aerialwaste.org/], a public dataset for illegal landfill discovery.

**Weights** Weights can be found and downloaded from this Google Drive [link](https://drive.google.com/drive/folders/1xy9BDFWWFkyaw3P8npEZxpTDFxkzA3NK?usp=sharing). In order to align the repository pulled repository with the code in `execute_model.ipynb`, create a `weights` folder to contain the downloaded `checkpoint.pth` file.

**Training details:** The model was trained using two GPUs Nvidia GeForce RTX 2080Ti and the following parameters:
- Learning rate: 0.005;
- Batch size: 12 (limited by the capacity of our server);
- Loss function: Binary Cross Entropy;
- Early stopping patience: 10;
- Early stopping min delta: 0.0005;
- Pretrained model: ImageNet pre-trained weights for ResNet50 backbone;
- Data augmentation: flip and rotation by multiples of 90°;
- Image size: 800. All images are resized to 800x800 during training;

Freezing the first two layers -the code of the backbone and the pre-trained weights can also be found in this repository-. 

A sigmoid function is applied over the last FC layer of the CNN to obtain the actual prediction of the model, that for the binary case of illegal landfills is a value between 0 and 1. 

**Execution:** The notebook `execute_model.ipynb` provides an example of how to load the trained model and execute it on an image to obtain the classification score as well as the CAMs. A couple of example images from Google Maps are provided to allow such execution attempts.

## License
Creative Commons CC BY licensing scheme (see LICENSE). 

## Cite us
```
@article{torres2023aerialwaste,
  title={AerialWaste dataset for landfill discovery in aerial and satellite images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={63},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
Visit our site for more details: https://aerialwaste.org/
