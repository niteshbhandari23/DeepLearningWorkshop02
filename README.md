# Cats vs Dogs vs Pandas Image Classification

### Name : Nitesh Bhandari K
### Reg.no :212225240101



## Project Structure

- `notebooks/cat_dog_panda_transfer_learning.ipynb`: Jupyter notebook containing the full data preparation, model training, and evaluation code.
- `requirements.txt`: List of dependencies required to run the project.
- `app.py`: Streamlit application for testing the model on new images.

## Setup Instructions

1. **Clone the repository:**
      ```bash
   git clone https://github.com/Jeyaarikaran/Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch.git
   cd Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Kaggle API Setup:**
   - To download the dataset, you need a Kaggle API token.
   - Go to your Kaggle account settings and click "Create New API Token".
   - Place the `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Your-Username>\.kaggle\` (Windows).

## CUDA Check

To ensure your environment is set up for GPU acceleration, you can run the following Python snippet:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

## Running the Notebook

Open the Jupyter notebook and run the cells to download the dataset, train the model, and view the evaluation results:

```bash
jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb
```

## Running the Streamlit App

To test the trained model on new images using a web interface:

```bash
streamlit run app.py
```

## Dataset

The dataset used in this project is the [Animal Image Dataset (Dog, Cat and Panda)](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) from Kaggle.

## Full Code

Here is the complete Python code for training the image classification model:

```python
import os
import shutil
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import kagglehub

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================
# 2. DATASET DOWNLOAD & PREPARATION
# ==========================================
# Download latest version
path = kagglehub.dataset_download("ashishsaxena2209/animal-image-datasetdog-cat-and-panda")
print("Path to downloaded dataset files:", path)

# Move data to ./data for compatibility
os.makedirs('./data', exist_ok=True)
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join('./data', item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# Safely find the correct data root (where class folders actually exist)
data_root = './data/animals/animals' 
if not os.path.exists(data_root):
    # Fallback in case folder structure differs
    data_root = './data'
print("Using data root:", data_root)

TRAIN = 'train'
TEST = 'test'

# ==========================================
# 3. TRANSFORMS & DATALOADERS
# ==========================================
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create two full datasets from the same root, but with different transforms
full_train_dataset = datasets.ImageFolder(data_root, data_transforms[TRAIN])
full_test_dataset = datasets.ImageFolder(data_root, data_transforms[TEST])

class_names = full_train_dataset.classes

# 80/20 Split
train_size = int(0.8 * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size

# Generate random indices for splitting consistently
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_indices)

image_datasets = {
    TRAIN: train_dataset,
    TEST: test_dataset
}

dataloaders = {
    TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN], batch_size=32, shuffle=True, num_workers=2),
    TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=32, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}

print("Classes:", class_names)
print("Dataset Sizes:", dataset_sizes)

# ==========================================
# 4. MODEL SETUP
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, len(class_names))
)

model = model.to(device)
print(model.fc)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Iterate over both TRAIN and TEST phases each epoch
        for phase in [TRAIN, TEST]:
            if phase == TRAIN:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward track history only if in train phase
                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model based on best TEST accuracy, not train accuracy
            if phase == TEST and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Acc: {best_acc:4f}')

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, criterion, optimizer, num_epochs=10)

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================
def evaluate_model(model):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders[TEST]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / dataset_sizes[TEST]
        test_acc = running_corrects.double() / dataset_sizes[TEST]

        print(f'\nFinal Test Loss: {test_loss:.4f} Final Test Acc: {test_acc:.4f}')

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

evaluate_model(model)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[TEST]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]} | true: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)

visualize_model(model)
```

## Expected Output :


#### Environment Setup :

<img width="1086" height="108" alt="image" src="https://github.com/user-attachments/assets/0a48322c-f282-4f13-b718-f3a6123a7d9a" />


#### Data Preparation :

<img width="1307" height="291" alt="image" src="https://github.com/user-attachments/assets/b3013c74-a54a-4ced-b49f-34d95aa2539b" />




<img width="972" height="78" alt="image" src="https://github.com/user-attachments/assets/fc80ed8c-0f65-493a-9bf2-0f0db0cb7ece" />


#### Model Design :

<img width="1292" height="187" alt="image" src="https://github.com/user-attachments/assets/00f81df5-c4ba-4570-b92d-70d68708f13f" />



#### Training :

<img width="1290" height="661" alt="image" src="https://github.com/user-attachments/assets/a7aed962-6f62-4441-b463-c889695b9d01" />


#### Evaluation Matrix :

<img width="1257" height="527" alt="image" src="https://github.com/user-attachments/assets/6a22ff22-53bf-4ac1-83b5-feeb32813572" />

#### Predicted Output :

<img width="1291" height="877" alt="image" src="https://github.com/user-attachments/assets/54696f87-ab1e-4ed2-9f36-b49f73a77af8" />






