import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler



def train(model,
          images_training,
          labels_training,
          images_validation,
          labels_validation,
          loss_func,
          batch_size,  
          learning_rate,
          epochs,
          model_validation=False,
          cuda_available=True,
          test_datalodaer=None,
          start_epoch=0,  # 修正参数名
          path_model='models/'):
    """
    train - Train the instance of the neural network.
    Args:
        model (torch): the instance of the neural network
        images_training, labels_training (numpy): the augmented training dataset
        images_validation, labels_validation (numpy): the resized validation dataset
        loss_func (class): the loss function
        start_epoch (int): the epoch to start train.ing from (supports resuming)
        batch_size (int): the number of samples per batch to load (default: 8)
        learning_rate (float): the learning rate (default: 1e-3)
        epochs (int): the total number of epochs to train (default: 80)
        model_validation (bool): whether to perform validation (default: False)
        cuda_available (bool): whether CUDA is available (default: True)
    """
    learning_rate = 1e-3

    # Use torch.utils.data to create a training_generator
    images_training = torch.Tensor(images_training)
    # images_training = images_training[0:10]
    labels_training = torch.Tensor(labels_training)
    # labels_training = labels_training[0:10]
    training_set = TensorDataset(images_training, labels_training)
    training_generator = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    # print(len(training_generator))

    # Use torch.utils.data to create a validation_generator
    if model_validation and len(images_validation) > 0:
        images_validation = torch.Tensor(images_validation)
        labels_validation = torch.Tensor(labels_validation)
        validation_set = TensorDataset(images_validation, labels_validation)
        validation_generator = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    # Implement Adam algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)


    # Loop over epochs
    for epoch in tqdm(range(start_epoch, epochs)):
        # Training
        print(f'\n---------Training for Epoch {epoch + 1} starting:---------')
        model.train()
        loss_training = 0
        # Loop over batches in an epoch using training_generator
        for index, (inputs, labels) in enumerate(training_generator):
            if cuda_available:
                # Transfer to GPU
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f'outputs.shape={outputs.shape}, labels.shape={labels.shape}')
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_training += loss

            if index % 20 == 0:
                loss_item = loss.item()
                print(f'→ Running_loss for Batch {index + 1}: {loss_item}')
        
        print(f'\033[1mTraining loss for Epoch {epoch + 1}: {loss_training}\033[0m\n')

        if model_validation and len(images_validation) > 0:
            # Validation
            print(f'--------Validation for Epoch {epoch + 1} starting:--------')
            model.eval()
            with torch.no_grad():
                loss_validation = 0
                # Loop over batches in an epoch using validation_generator
                for index, (inputs, labels) in enumerate(validation_generator):
                    if cuda_available:
                        # Transfer to GPU
                        inputs, labels = inputs.cuda(), labels.cuda()
                
                    outputs = model(inputs)
                    loss_validation += loss_func(outputs, labels)
                
            print(f'\033[1mValidation loss for Epoch {epoch + 1}: {loss_validation}\033[0m\n')

        scheduler.step()
                    
        # Save model for the current epoch
        save_path = os.path.join(path_model, f'model_epoch_{epoch + 1}_{type(model).__name__}.pth')
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func,
                }, save_path)

        print(f"Model saved at {save_path}")
