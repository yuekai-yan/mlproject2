# Process the training dataset
training_data_processing = True
# Train the model
model_training = True
# Validation the model
model_validation = False
# Load the model from your Google Drive or local file system
model_loading = False

import os
import re
import numpy as np
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from skimage.transform import resize
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

from training_data_processing import *
from loss_function import *
from training import *
from testing_data_processing import *

from networks.LinkNetB7 import *
from networks.DLinkNet34 import *
from networks.DLinkNet50 import *
from networks.DLinkNet101 import *
from networks.LinkNet34 import *
from networks.UNet import *
from networks.TransUNet import TransUNetTrainer, create_dataloaders

path_training = 'training/'
path_testing = 'test_set_images/'
path_data = 'data/'
path_model = 'models/'

"""
Get Device for Training
"""
# Determine if your system supports CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    print('CUDA is available. Utilize GPUs for computation')
    device = torch.device("cuda")
else:
    print('CUDA is not available. Utilize CPUs for computation.')
    device = torch.device("cpu")


"""
Load and Process the Training Dataset
"""
# The resolution of resized training images and the corresponding masks
training_resize = 384
# The number of resized training pairs used for data augmentation
training_number = 100
# The resolution of resized testing images
testing_resize = int(608 * training_resize / 400)
if testing_resize % 2 == 1:
    testing_resize += 1

if not model_loading:
    # Load the augmented training dataset and resized validation dataset
    images_augmented = np.load(f'{path_data}images_training.npy')
    labels_augmented = np.load(f'{path_data}labels_training.npy')
    images_validation = np.load(f'{path_data}images_validation.npy')
    labels_validation = np.load(f'{path_data}labels_validation.npy')

    images_augmented = torch.tensor(images_augmented, dtype=torch.float32).to(device)
    labels_augmented = torch.tensor(labels_augmented, dtype=torch.float32).to(device)
    images_validation = torch.tensor(images_validation, dtype=torch.float32).to(device)
    labels_validation = torch.tensor(labels_validation, dtype=torch.float32).to(device)
elif training_data_processing:
    # Load and generate the resized training dataset and validation dataset
    images_training, labels_training, images_validation, labels_validation = training_data_loading(
        path_training, training_resize, training_number)
    # Generate the augmented training dataset
    rotations = [0, 45, 90, 135]  # the rotation angle
    flips = ['original', np.flipud, np.fliplr]  # 'original', np.flipud, np.fliplr
    shifts = [(-16, 16)]
    images_augmented, labels_augmented = training_data_augmentation(
        images_training, labels_training, rotations, flips, shifts, training_resize)
    # Save the augmented training dataset and resized validation dataset
    np.save(f'{path_data}images_training', images_augmented)
    np.save(f'{path_data}labels_training', labels_augmented)
    np.save(f'{path_data}images_validation', images_validation)
    np.save(f'{path_data}labels_validation', labels_validation)
"""
Create Instances of Neural Networks for Ensemble Learning
"""
# Initialize multiple models for ensemble
models = [
    # DLinkNet34().to(device), # 80
    # DLinkNet50().to(device), # 1
    # DLinkNet101().to(device), # scratch
    # LinkNet34().to(device), # 80
    # LinkNetB7().to(device), # 35
    # UNet().to(device) # 11
    TransUNetTrainer(img_size=training_resize, num_classes=2, config_name="ViT-B_16").to(device)
]



def get_max_epoch_model_file(path_model, model_name):
    max_epoch = -1
    best_model_file = None
    pattern = re.compile(rf"model_epoch_(\d+)_{model_name}\.pth")
    
    for file in os.listdir(path_model):
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_model_file = file
    
    return max_epoch, best_model_file

# Optionally load pre-trained weights for each model
def get_max_epoch_model_file(path_model, model_name):
    max_epoch = -1
    best_model_file = None
    pattern = re.compile(rf"model_epoch_(\d+)_{model_name}\.pth")
    
    for file in os.listdir(path_model):
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_model_file = file
    
    return max_epoch, best_model_file


# # Initialize TransUNet Trainer
# trainer = TransUNetTrainer(config_name="ViT-B_16", img_size=384, num_classes=2, lr=1e-3, batch_size=64, max_epochs=100)
# print(images_augmented.shape)

# # Create dataloaders
# train_loader, val_loader = create_dataloaders(images_augmented, labels_augmented, images_validation, labels_validation, batch_size=8)

# # Train the model
# trainer.train(train_loader, val_loader)



for model in models:
    model_name = type(model).__name__
    start_max_epoch, best_model_file = get_max_epoch_model_file(path_model, model_name)
    
    if best_model_file:
        checkpoint_path = os.path.join(path_model, best_model_file)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights for {model_name} from epoch {start_max_epoch}")
    else:
        print(f"No pre-trained weights found for {model_name}. Training from scratch.")


    """
    Train Each Model in the Ensemble
    """
    print(f"\nTraining model {type(model).__name__}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)


    train(model,
            images_augmented,
            labels_augmented,
            images_validation,
            labels_validation,
            loss_func=BCEIoULoss(),  # BCEIoULoss(), DiceBCELoss(), nn.BCELoss()
            batch_size=16,
            learning_rate=1e-3,
            start_epoch=start_max_epoch,
            epochs=80,
            model_validation=model_validation,
            cuda_available=cuda_available,
            path_model=path_model)

    """
    Process the Testing Dataset and Create the Submission File
    """
    submission = submission_creating(model,
                                    path_testing,
                                    training_resize,
                                    testing_resize,
                                    cuda_available)
    print(submission)


    np.savetxt(f'submit_{model_name}_{start_max_epoch}.csv', submission, delimiter=",", fmt='%s')


# submission = submission_creating_ensemble(models,
#                                     path_testing,
#                                     training_resize,
#                                     testing_resize,
#                                     cuda_available)

# np.savetxt(f'submit_ensemble_{start_max_epoch}.csv', submission, delimiter=",", fmt='%s')
