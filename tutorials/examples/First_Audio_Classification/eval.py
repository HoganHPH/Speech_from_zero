import argparse
import torch

from model import AudioClassifier
from datasets import SoundDataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='./DATA', help="data dir")
parser.add_argument('-pt', '--pretrained', type=str, default="model.pt", help="name of pretrained model")
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="number of samples in each batch")
parser.add_argument('-dv', '--device', type=str, default='cuda', help="cuda or cpu")
args = parser.parse_args()


def eval(model, test_dl, device):
    correct_prediction = 0
    total_prediction = 0
    
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Normalize inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            # Get predictions
            outputs = model(inputs)
            
            _, prediction = torch.max(outputs, 1)
            crrect_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            
        acc = correct_prediction / total_prediction
        print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
        
        
if __name__ == '__main__':
    
    data_path = args.data
    pretrained = args.pretrained
    batch_size = args.batch_size 
    
    device = args.device
    device = torch.device(device)
    
    loader = SoundDataLoader(data_path=data_path, batch_size=batch_size, phase='test')
    test_dl = loader.load_data()
    
    model = AudioClassifier()
    model = model.to(device)
    model.load_state_dict(torch.load(pretrained))
    model.eval()
    
    eval(model=model, test_dl=test_dl, device=device)
    
    