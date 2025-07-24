import numpy as np
import torch
from tqdm import tqdm

def random_seed_initialization():
  seed_value = 42
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)

def train_loop(model, optimizer, criterion, train_loader, device):
  train_loss = []
  train_accuracy = []
  model.train()
  model.to(device)
  for batch in tqdm(train_loader):
    x, y = batch
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    train_loss.append(loss.detach().cpu().item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_accuracy.append(100*torch.mean((torch.argmax(y_pred,dim=1) == y).float()).item())

  train_loss = np.mean(train_loss)
  train_accuracy = np.mean(train_accuracy)

  return train_loss, train_accuracy

def validation_loop(model, criterion, val_loader, device):
  validation_loss = []
  validation_accuracy = []

  model.eval()
  model.to(device)
  with torch.no_grad():
    for batch in tqdm(val_loader):
      x, y = batch
      x, y = x.to(device), y.to(device)
      y_val_pred = model(x)
      val_loss = criterion(y_val_pred, y)
      validation_loss.append(val_loss.detach().cpu().item())
      validation_accuracy.append(100*torch.mean((torch.argmax(y_val_pred,dim=1) == y).float()).item())


  validation_loss = np.mean(validation_loss)
  validation_accuracy = np.mean(validation_accuracy)
  return validation_loss, validation_accuracy