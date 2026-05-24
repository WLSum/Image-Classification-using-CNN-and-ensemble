# Image-Classification-using-CNN-and-ensemble

This project focuses on scene classification using convolutional neural networks (CNNs). The proposed system integrates several advanced components to enhance performance, including the Convolutional Block Attention Module (CBAM) for attention-based feature refinement, Spatial Pyramid Pooling (SPP) for multi-scale feature extraction, residual connections to improve optimization, and partial replacement of fully connected layers with Global Average Pooling (GAP) for improved efficiency. Additionally, an ensemble approach is employed to further boost classification accuracy. The overall architecture draws inspiration from recent works by Yee et al. (2022), Woo et al. (2018) and He et al. (2015).

This project uses data from [Scene-15](https://www.kaggle.com/datasets/yiklunchow/scene15) to train and test the model. The model achieve an accuracy of 92.84% on the test set.


# To run the model
1. Download the dataset from Kaggle, and install the required libraries

```bash
pip install torch torchvision scikit-learn
```


2. Sample code to train the model on train images
```bash
python scene_recog_cnn.py --phase train --train_data_dir ./data/train --model_dir trained_cnn.pth
```


3. Sample code to run the model on test images

```bash
python scene_recog_cnn.py --phase test --test_data_dir ./data/test --model_dir trained_cnn.pth
```
