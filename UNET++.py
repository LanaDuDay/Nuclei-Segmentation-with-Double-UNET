#pip install segmentation-models-pytorch

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from sklearn.metrics import jaccard_score

# Hàm tính IoU
def calculate_iou(pred_mask, true_mask):
    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    true_mask = true_mask.cpu().numpy().astype(np.uint8)
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        self.image_list = os.listdir(image_folder)
        self.mask_list = os.listdir(mask_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_list[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_list[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        return image, mask

# Đường dẫn đến thư mục chứa ảnh và mask
train_image_path = r'/content/drive/MyDrive/images'
mask_image_path = r'/content/drive/MyDrive/masks'

# Định nghĩa các phép augmentation
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_dataset = CustomDataset(image_folder=train_image_path, mask_folder=mask_image_path, transform=transform)

# DataLoader để tải dữ liệu theo batch
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Chọn mô hình
model = UnetPlusPlus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=1).to(device)

# Chọn hàm mất mát và tối ưu hóa
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_iou = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Tính IoU
        outputs_thresholded = (outputs > 0.5).float()
        iou = calculate_iou(outputs_thresholded, masks)
        epoch_iou += iou

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, IoU: {epoch_iou/len(train_loader):.4f}")

# Lưu mô hình đã huấn luyện
torch.save(model.state_dict(), "unetplusplus_model.pth")

class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        self.image_list = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_list[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = transforms.ToTensor()(image)

        return image

# Đường dẫn đến thư mục chứa ảnh mới (không có mask)
new_images_path = r'/content/drive/MyDrive/test'

# Tạo một TestDataset cho dữ liệu mới
test_dataset = TestDataset(image_folder=new_images_path, transform=A.Resize(256, 256))

# DataLoader để tải dữ liệu mới theo batch
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Chạy dự đoán trên dữ liệu mới
model.eval()  # Chuyển sang chế độ đánh giá (không cần tính gradient)

# Lặp qua các ảnh dự đoán và tạo contour
for i, new_images in enumerate(test_loader):
    outputs = model(new_images.to(device))  # Move input data to GPU

    for j in range(outputs.shape[0]):
        predicted_mask = outputs[j, 0].detach().cpu().numpy()

        # Chuyển đổi ảnh binary sang ảnh grayscale
        _, binary_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)

        # Áp dụng morphological opening để loại bỏ nhiễu
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Trích xuất contour từ ảnh binary
        contours, _ = cv2.findContours(np.uint8(binary_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc các contour dựa trên diện tích
        min_area = 50  # Điều chỉnh giá trị này tùy thuộc vào kích thước tế bào
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Tạo một bản sao của ảnh gốc để vẽ contour lên đó
        original_image_copy = np.transpose(new_images[j].numpy(), (1, 2, 0)).copy()

        # Vẽ contour đã lọc trên bản sao của ảnh gốc
        cv2.drawContours(original_image_copy, filtered_contours, -1, (0, 255, 0), 1)

        # Lưu ảnh chứa contour
        output_filename = f"output_image_{i * batch_size + j}.jpg"
        cv2.imwrite(output_filename, original_image_copy)
