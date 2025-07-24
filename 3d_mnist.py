# Reference codebase: https://www.kaggle.com/code/shivamb/3d-convolutions-understanding-use-case 

import torch
import h5py
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from model_file import MNIST_3D
import wandb
from utils import random_seed_initialization, train_loop, validation_loop
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm, trange
import os

class Configuration:
    batch_size = 32
    learning_rate = 0.001
    num_classes = 10
    epochs = 30
    record_plots = True
    model_file_path_dir = ".//Model_weights//"

def add_rgb_dimention(array):
    scaler_map = plt.cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, : -1]
    return array

def main():

    wandb.login()

    random_seed_initialization()

    cfg = Configuration
    with h5py.File('full_dataset_vectors.h5', 'r') as dataset:
        x_train = dataset["X_train"][:]
        x_val = dataset["X_test"][:]
        y_train = dataset["y_train"][:]
        y_val = dataset["y_test"][:]

    #print ("x_train shape: ", x_train.shape)
    #print ("y_train shape: ", y_train.shape)

    #print ("x_val shape:  ", x_val.shape)
    #print ("y_val shape:  ", y_val.shape)

    #print(f"Unique elements : {np.unique(y_train)}")

    #element_count_train = Counter(y_train).most_common()
    #element_count_test = Counter(y_val).most_common()

    #print(element_count_test)
    #print(element_count_train)

    class_indices = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=class_indices, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    xtrain = np.ndarray((x_train.shape[0], 4096, 3))
    xval = np.ndarray((x_val.shape[0], 4096, 3))

    for i in range(x_train.shape[0]):
        xtrain[i] = add_rgb_dimention(x_train[i])
    for i in range(x_val.shape[0]):
        xval[i] = add_rgb_dimention(x_val[i])

    x_train = xtrain.reshape(x_train.shape[0], 3, 16, 16, 16)
    x_val = xval.reshape(x_val.shape[0], 3, 16, 16, 16)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    x_train = x_train.type(torch.float32)
    y_train = y_train.type(torch.LongTensor)
    x_val = x_val.type(torch.float32)
    y_val = y_val.type(torch.LongTensor)

    print(f"The shape of x_train : {x_train.shape}")
    print(f"The shape of y_train : {y_train.shape}")
    print(f"The shape of x_val : {x_val.shape}")
    print(f"The shape of y_val : {y_val.shape}")

    model = MNIST_3D(num_classes=cfg.num_classes)

    
    train_dataset = TensorDataset(x_train, y_train)
    validation_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(validation_dataset, cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.learning_rate)


    if (cfg.record_plots):
        wandb.init(
        project="3D_MNIST",
        name=f"3d_mnist",
        reinit=True,
        config={
        "learning_rate": cfg.learning_rate,
        "architecture": "Simple_CNN",
        "dataset": "3D_MNIST",
        "epochs": cfg.epochs,
        })

    best_validation_accuracy = 0

    for epoch in trange(cfg.epochs, desc="Training"):

            # Train loop
            train_loss, train_accuracy = train_loop(model, optimizer, criterion, train_loader, device)

            # Validation loop
            validation_loss, validation_accuracy = validation_loop(model, criterion, val_loader, device)

            print(f"Training Loss : {train_loss:.2f}, Training Accuracy : {train_accuracy:.2f}, Validation Loss : {validation_loss:.2f}, Validation Accuracy : {validation_accuracy:.2f}")
            if (cfg.record_plots):
                wandb.log({"Training_loss": train_loss, "Training_accuracy": train_accuracy, "Validation_loss": validation_loss, "Validation_accuracy": validation_accuracy})

            if (validation_accuracy > best_validation_accuracy):
                best_validation_accuracy = validation_accuracy
                torch.save(model.state_dict(), os.path.join(cfg.model_file_path_dir, str(cfg.num_classes) + "_classes_val_accuracy_" + str(int(validation_accuracy)) + ".pt"))
            
            #scheduler.step()
        
    wandb.finish()



    

if __name__ == "__main__":
    main()