# Nuclei-Segmentation-with-Double-UNET
This repository contains the implementation of UNet++ architecture for processing and segmenting medical images in the Kaggle Data Science Bowl 2018 competition. The dataset has been pre-processed to facilitate easy training and evaluation of the model.
Paper: https://arxiv.org/abs/1807.10165
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Overview
UNet++ is a state-of-the-art neural network architecture designed for biomedical image segmentation. It builds on the original UNet architecture, incorporating nested and dense skip pathways to improve segmentation accuracy.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Features
- Pre-processed Dataset: The repository includes pre-processed data from the Kaggle Data Science Bowl 2018, making it easy to get started with training the UNet++ model. Here is the link to the dataset: https://www.kaggle.com/datasets/84613660e1f97d3b23a89deb1ae6199a0c795ec1f31e2934527a7f7aad7d8c37
- UNet++ Implementation: A detailed and efficient implementation of the UNet++ architecture.
- Training and Evaluation Scripts: Scripts to train the model and evaluate its performance on the dataset.
- Customizable Hyperparameters: Easily adjust hyperparameters to experiment and optimize the model.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Model Outputs:
## Training Results

Using device: cuda

| Epoch | Loss   | IoU    |
|-------|--------|--------|
| 1/10  | 0.0825 | 0.9167 |
| 2/10  | 0.0115 | 1.0000 |
| 3/10  | 0.0080 | 1.0000 |
| 4/10  | 0.0070 | 1.0000 |
| 5/10  | 0.0066 | 1.0000 |
| 6/10  | 0.0064 | 1.0000 |
| 7/10  | 0.0063 | 1.0000 |
| 8/10  | 0.0063 | 1.0000 |
| 9/10  | 0.0063 | 1.0000 |
| 10/10 | 0.0062 | 1.0000 |

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Installation

```pip install segmentation-models-pytorch```

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dataset Preparation

Prepare your dataset by organizing the images and masks into separate directories. Ensure the images and masks are properly preprocessed and saved in the following structure:

/content/drive/MyDrive/

    ├── images/
    
    └── masks/

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Usage
Data Augmentation and Custom Dataset
The CustomDataset class handles loading and transforming the images and masks. Augmentation is performed using the albumentations library.

````class CustomDataset(Dataset):
    # Your CustomDataset implementation
````

Define Augmentations
````
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
````

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model Training
DataLoader
Prepare the DataLoader to load data in batches:

````
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
````
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model, Loss, and Optimizer
Define the UNet++ model, loss function, and optimizer:

````
model = UnetPlusPlus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
````

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Training Loop
Train the model using the training loop:

````
for epoch in range(num_epochs):
    # Your training loop implementation
````
Save the trained model:

````
torch.save(model.state_dict(), "unetplusplus_model.pth")
````

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Inference
Test Dataset and DataLoader
Prepare the TestDataset class for inference on new images:

````
class TestDataset(Dataset):
    # Your TestDataset implementation
````
Load test data:

````
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
````
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Predict and Save Results
Run inference and save the output images with contours:

````
for i, new_images in enumerate(test_loader):
    # Your inference implementation
````
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Results
Sample results and performance metrics are stored in the results directory. The trained model is saved in the models directory.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Acknowledgements

- UNet++: A Nested U-Net Architecture for Medical Image Segmentation
- Kaggle Data Science Bowl 2018 organizers and participants.

Feel free to modify any sections to better fit your repository.
