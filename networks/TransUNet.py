import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from loss_function import *
from torchvision import transforms
sys.path.append(os.path.dirname(__file__)+'/../networks/')

from networks.vit_seg_modeling import VisionTransformer, CONFIGS
# from utils import DiceLoss, calculate_metric_percase
from vit_seg_configs import get_b16_config



class TransUNetTrainer(nn.Module):
    def __init__(self, img_size, num_classes, config_name):
        super(TransUNetTrainer, self).__init__()
        # Configuration and model initialization
        self.config = CONFIGS[config_name]
        self.config.n_classes = num_classes
        self.img_size = img_size
        self.model = VisionTransformer(self.config, img_size=self.img_size, num_classes=num_classes).cuda()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        # self.batch_size = batch_size
        # self.max_epochs = max_epochs
        # self.BCEIoULoss = BCEIoULoss()
    
    def forward(self, x):
        x = self.model(x)
        return x

    # def train(self, train_loader, val_loader=None):
    #     print(f"Starting training for {self.max_epochs} epochs...")
    #     for epoch in tqdm(range(self.max_epochs), desc="Training Progress"):
    #         self.model.train()
    #         train_loss = 0.0
    #         for images, labels in train_loader:
    #             images, labels = images.cuda(), labels.cuda()

    #             self.optimizer.zero_grad()
    #             # print(f'images.shape={images.shape}')
    #             outputs = self.model(images)
    #             # print(f'outputs.shape={outputs.shape}')
    #             # print(f'labels.shape={labels.shape}')
    #             loss = self.BCEIoULoss(outputs, labels)
    #             print(f'loss={loss}')
    #             loss.backward()
    #             self.optimizer.step()
    #             train_loss += loss.item()

    #         print(f"Epoch {epoch + 1}/{self.max_epochs}, Training Loss: {train_loss:.4f}")
    #         self.scheduler.step()

    #         if val_loader:
    #             self.validate(val_loader)

    # def validate(self, val_loader):
    #     self.model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, labels in val_loader:
    #             images, labels = images.cuda(), labels.cuda()
    #             outputs = self.model(images)
    #             loss = self.criterion(outputs, labels, softmax=True)
    #             val_loss += loss.item()
    #     print(f"Validation Loss: {val_loss:.4f}")

    def save_model(self, path="transunet.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")

    def load_model(self, path="transunet.pth"):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def create_dataloaders(train_data, train_labels, batch_size, val_data=None, val_labels=None):
    print(f'creat_dataloader_train_data.shape={train_data.shape}')
    print(f'creat_dataloader_train_labels.shape={train_labels.shape}')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_data is not None and val_labels is not None:
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
