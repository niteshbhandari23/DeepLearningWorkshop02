import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cats vs Dogs vs Pandas Image Classification\n",
                "This notebook demonstrates how to build an image classification model using transfer learning in PyTorch."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torchvision\n",
                "import torchaudio\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import os\n",
                "import shutil\n",
                "import time\n",
                "import copy\n",
                "\n",
                "print(\"PyTorch Version: \", torch.__version__)\n",
                "print(\"Torchvision Version: \", torchvision.__version__)\n",
                "print(\"CUDA available:\", torch.cuda.is_available())\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(\"Device:\", device)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Preparation\n",
                "Download the dataset from Kaggle and prepare DataLoaders."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Note: Ensure you have your kaggle.json configured.\n",
                "!pip install kaggle\n",
                "!kaggle datasets download -d gpiosenka/cats-dogs-pandas-images -p ./data --unzip"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from torchvision import datasets, transforms\n",
                "\n",
                "# The dataset might have train, valid, test folders.\n",
                "# Assuming it's downloaded into ./data\n",
                "data_dir = './data'\n",
                "TRAIN = 'train'\n",
                "TEST = 'test'\n",
                "\n",
                "# Transforms\n",
                "data_transforms = {\n",
                "    TRAIN: transforms.Compose([\n",
                "        transforms.RandomResizedCrop(224),\n",
                "        transforms.RandomHorizontalFlip(),\n",
                "        transforms.RandomRotation(15),\n",
                "        transforms.ToTensor(),\n",
                "        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
                "    ]),\n",
                "    TEST: transforms.Compose([\n",
                "        transforms.Resize(256),\n",
                "        transforms.CenterCrop(224),\n",
                "        transforms.ToTensor(),\n",
                "        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
                "    ]),\n",
                "}\n",
                "\n",
                "image_datasets = {\n",
                "    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])\n",
                "    for x in [TRAIN, TEST] if os.path.exists(os.path.join(data_dir, x))\n",
                "}\n",
                "\n",
                "dataloaders = {\n",
                "    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)\n",
                "    for x in image_datasets.keys()\n",
                "}\n",
                "\n",
                "dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}\n",
                "class_names = image_datasets[TRAIN].classes\n",
                "\n",
                "print(\"Classes:\", class_names)\n",
                "print(\"Dataset Sizes:\", dataset_sizes)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Model Design\n",
                "Load ResNet18, freeze convolutional layers, and update classifier head."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch.nn as nn\n",
                "from torchvision import models\n",
                "\n",
                "model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)\n",
                "\n",
                "# Freeze convolutional layers\n",
                "for param in model.parameters():\n",
                "    param.requires_grad = False\n",
                "\n",
                "# Replace classifier head\n",
                "num_ftrs = model.fc.in_features\n",
                "model.fc = nn.Sequential(\n",
                "    nn.Linear(num_ftrs, 256),\n",
                "    nn.ReLU(),\n",
                "    nn.Dropout(p=0.5),\n",
                "    nn.Linear(256, len(class_names))\n",
                ")\n",
                "\n",
                "model = model.to(device)\n",
                "print(model.fc)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Training"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch.optim as optim\n",
                "import random\n",
                "\n",
                "# Set random seeds for reproducibility\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)\n",
                "random.seed(42)\n",
                "\n",
                "criterion = nn.CrossEntropyLoss()\n",
                "optimizer = optim.Adam(model.fc.parameters(), lr=0.001)\n",
                "\n",
                "def train_model(model, criterion, optimizer, num_epochs=10):\n",
                "    since = time.time()\n",
                "    best_model_wts = copy.deepcopy(model.state_dict())\n",
                "    best_acc = 0.0\n",
                "\n",
                "    for epoch in range(num_epochs):\n",
                "        print(f'Epoch {epoch}/{num_epochs - 1}')\n",
                "        print('-' * 10)\n",
                "\n",
                "        # We only train on 'train' since 'val' is not explicitly requested, but we could evaluate on 'test' during training if needed.\n",
                "        # Usually we evaluate on validation set, let's just train here and eval later.\n",
                "        model.train()\n",
                "        running_loss = 0.0\n",
                "        running_corrects = 0\n",
                "\n",
                "        for inputs, labels in dataloaders[TRAIN]:\n",
                "            inputs = inputs.to(device)\n",
                "            labels = labels.to(device)\n",
                "\n",
                "            optimizer.zero_grad()\n",
                "\n",
                "            with torch.set_grad_enabled(True):\n",
                "                outputs = model(inputs)\n",
                "                _, preds = torch.max(outputs, 1)\n",
                "                loss = criterion(outputs, labels)\n",
                "\n",
                "                loss.backward()\n",
                "                optimizer.step()\n",
                "\n",
                "            running_loss += loss.item() * inputs.size(0)\n",
                "            running_corrects += torch.sum(preds == labels.data)\n",
                "\n",
                "        epoch_loss = running_loss / dataset_sizes[TRAIN]\n",
                "        epoch_acc = running_corrects.double() / dataset_sizes[TRAIN]\n",
                "\n",
                "        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')\n",
                "        \n",
                "        # Save best model simply based on train acc as there is no valid set by default.\n",
                "        if epoch_acc > best_acc:\n",
                "            best_acc = epoch_acc\n",
                "            best_model_wts = copy.deepcopy(model.state_dict())\n",
                "            torch.save(model.state_dict(), 'best_model.pth')\n",
                "\n",
                "    time_elapsed = time.time() - since\n",
                "    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')\n",
                "    print(f'Best Acc: {best_acc:4f}')\n",
                "\n",
                "    model.load_state_dict(best_model_wts)\n",
                "    return model\n",
                "\n",
                "model = train_model(model, criterion, optimizer, num_epochs=10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
                "\n",
                "def evaluate_model(model):\n",
                "    model.eval()\n",
                "    running_loss = 0.0\n",
                "    running_corrects = 0\n",
                "    \n",
                "    all_preds = []\n",
                "    all_labels = []\n",
                "\n",
                "    with torch.no_grad():\n",
                "        for inputs, labels in dataloaders[TEST]:\n",
                "            inputs = inputs.to(device)\n",
                "            labels = labels.to(device)\n",
                "\n",
                "            outputs = model(inputs)\n",
                "            _, preds = torch.max(outputs, 1)\n",
                "            loss = criterion(outputs, labels)\n",
                "\n",
                "            running_loss += loss.item() * inputs.size(0)\n",
                "            running_corrects += torch.sum(preds == labels.data)\n",
                "            \n",
                "            all_preds.extend(preds.cpu().numpy())\n",
                "            all_labels.extend(labels.cpu().numpy())\n",
                "\n",
                "        test_loss = running_loss / dataset_sizes[TEST]\n",
                "        test_acc = running_corrects.double() / dataset_sizes[TEST]\n",
                "\n",
                "        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')\n",
                "        \n",
                "        # Confusion Matrix\n",
                "        cm = confusion_matrix(all_labels, all_preds)\n",
                "        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\n",
                "        disp.plot(cmap=plt.cm.Blues)\n",
                "        plt.show()\n",
                "\n",
                "evaluate_model(model)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def imshow(inp, title=None):\n",
                "    inp = inp.numpy().transpose((1, 2, 0))\n",
                "    mean = np.array([0.485, 0.456, 0.406])\n",
                "    std = np.array([0.229, 0.224, 0.225])\n",
                "    inp = std * inp + mean\n",
                "    inp = np.clip(inp, 0, 1)\n",
                "    plt.imshow(inp)\n",
                "    if title is not None:\n",
                "        plt.title(title)\n",
                "    plt.pause(0.001)\n",
                "\n",
                "def visualize_model(model, num_images=6):\n",
                "    was_training = model.training\n",
                "    model.eval()\n",
                "    images_so_far = 0\n",
                "    fig = plt.figure()\n",
                "\n",
                "    with torch.no_grad():\n",
                "        for i, (inputs, labels) in enumerate(dataloaders[TEST]):\n",
                "            inputs = inputs.to(device)\n",
                "            labels = labels.to(device)\n",
                "\n",
                "            outputs = model(inputs)\n",
                "            _, preds = torch.max(outputs, 1)\n",
                "\n",
                "            for j in range(inputs.size()[0]):\n",
                "                images_so_far += 1\n",
                "                ax = plt.subplot(num_images//2, 2, images_so_far)\n",
                "                ax.axis('off')\n",
                "                ax.set_title(f'predicted: {class_names[preds[j]]} | true: {class_names[labels[j]]}')\n",
                "                imshow(inputs.cpu().data[j])\n",
                "\n",
                "                if images_so_far == num_images:\n",
                "                    model.train(mode=was_training)\n",
                "                    return\n",
                "        model.train(mode=was_training)\n",
                "\n",
                "visualize_model(model)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/cat_dog_panda_transfer_learning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated.")
