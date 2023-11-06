import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)

class EasyImageClassifier:
  def __init__(self, data=os.path.join(src_dir, "easy_image_classifier" 'data'), model_type='resnet101', num_epochs=25, lr=0.001, step_size=7, gamma=0.1, momentum=0.9):
    self.data = data
    self.model_type = model_type
    self.num_epochs = num_epochs
    self.lr = lr
    self.step_size = step_size
    self.gamma = gamma
    self.momentum = momentum

    self.mean = np.array([0.5, 0.5, 0.5])
    self.std = np.array([0.25, 0.25, 0.25])

    self.data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]),
    }

    self.data_dir = data
    self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'val']}
    self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'val']}
    self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
    self.class_names = self.image_datasets['train'].classes

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Class names: ", ", ".join(self.class_names))

  def display_rand(self):
    inputs, classes = next(iter(self.dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)

    inp = out.numpy().transpose((1, 2, 0))
    inp = self.std * inp + self.mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title([self.class_names[x] for x in classes])
    plt.show()

  def _train_model(self, model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

  def generate_model(self):
    if self.model_type == 'resnet18':
        model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    elif self.model_type == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif self.model_type == 'resnet101':
        model = models.resnet101(weights="IMAGENET1K_V2")
    elif self.model_type == 'resnet152':
        model = models.resnet152(weights="IMAGENET1K_V2")
    else:
        raise ValueError("Invalid model type")

    for param in model.parameters():
      param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(self.class_names))

    model = model.to(self.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.fc.parameters(), lr=self.lr, momentum=self.momentum)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    model = self._train_model(model, criterion, optimizer,
                            exp_lr_scheduler, num_epochs=self.num_epochs)
    return model

  def save_model(self, model, save_path):
    torch.save(model, save_path)

  def predict_image(self, model, image=os.path.join(src_dir, "easy_image_classifier", 'data', 'predict', "oranges-in-a-box.jpg")):
    model = torch.load(model, map_location=self.device).to(self.device)
    model.eval()

    img = Image.open(image).convert('RGB')
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    transform_tensor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    img_y = transform_tensor(img).unsqueeze(0).float().to(self.device)

    prediction = torch.argmax(model(img_y))

    predicted_class = self.class_names[prediction]

    print("Prediction: ", predicted_class)

  def predict_images(self, model, images=os.path.join(src_dir, "easy_image_classifier" 'data', 'path')):
    model = torch.load(model, map_location=self.device).to(self.device)
    model.eval()

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    transform_tensor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    for filename in os.listdir(images):
      f = os.path.join(images, filename)

      if os.path.isfile(f):
        img = Image.open(f).convert('RGB')

        img_y = transform_tensor(img).unsqueeze(0).float().to(self.device)

        prediction = torch.argmax(model(img_y))

        predicted_class = self.class_names[prediction]

        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        plt.title(f'Prediction: {predicted_class}')
        plt.axis('off')
        plt.show()
