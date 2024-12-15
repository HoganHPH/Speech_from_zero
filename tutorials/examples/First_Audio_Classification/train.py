import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from model import AudioClassifier
from datasets import SoundDataLoader
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='./DATA', help="data dir")
parser.add_argument('-bs', '--batch_size', type=int, default=16, help="number of samples in each batch")
parser.add_argument('-ep', '--epochs', type=int, default=100, help="number of training epochs")
parser.add_argument('-dv', '--device', type=str, default='cuda', help="cuda or cpu")
args = parser.parse_args()


def train(model, train_dl, num_epochs, device):
    # Tensorboard
    writer = SummaryWriter()
    
    # Loss function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy='linear'
    )
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        print(f"\n====> Epoch {epoch + 1}:\n")
        for i, data in tqdm(enumerate(train_dl), total=len(train_dl)):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Normalize inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        avg_acc = correct_prediction / total_prediction
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/train", avg_acc, epoch)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}')
        
    torch.save(model.state_dict(), 'model.pt')
    print("Finished Training")
    

if __name__ == '__main__':
    
    data_path = args.data
    num_epochs = args.epochs
    batch_size = args.batch_size 
    device = args.device
    
    device = torch.device(device)
    print("Device : ", device)
    
    # Create the model and put it on the GPU if available
    model = AudioClassifier()
    model = model.to(device)   
    
    loader = SoundDataLoader(data_path=data_path, batch_size=batch_size, phase='train')
    train_dl = loader.load_data()
    train(model=model, train_dl=train_dl, num_epochs=num_epochs, device=device)