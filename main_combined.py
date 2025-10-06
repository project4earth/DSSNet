import warnings
import torch
import argparse
import torch.optim as optim
from datetime import datetime
from model.dssnet import DSSNet
from train_combined import ModelTrainer
from test_combined import ModelTester
from dataset_combined import get_dataloaders
from utils import config
from loss_function import TverskyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchprofile.profile")
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def initialize_model(model_type, in_channels_1, in_channels_2, num_classes):
    model_dict = {
        'dssnet': DSSNet(in_channels_1=in_channels_1, in_channels_2=in_channels_2, num_classes=num_classes),
    } 

    if model_type not in model_dict:
        raise ValueError("Invalid model type.")
    
    model = model_dict[model_type]

    return model

def setup_training(model, train_loader, val_loader, num_epochs, checkpoint_path, log_filename, device, resume=None):   
    loss_function = TverskyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0, T_mult=config.T_MULT, eta_min=config.ETA_MIN)  
    trainer = ModelTrainer(model, train_loader, val_loader, loss_function, optimizer, scheduler,
                           num_epochs, config.PATIENCE, checkpoint_path, log_filename, device, resume)
    
    return trainer

def main(args):
    print(f'Preparing data and model...')

    train_loader, val_loader = get_dataloaders(data_root=config.ROOT_DIR, mode='training') # type: ignore
    test_loader = get_dataloaders(data_root=config.ROOT_DIR, mode='testing')

    model_types = ['dssnet']

    for model_type in model_types:
        print(f'Using model: {model_type}')

        model = initialize_model(model_type=model_type, in_channels_1=2, in_channels_2=4, num_classes=2)
        model = model.to(device)

        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = f'{model_type}_{current_datetime}'
        checkpoint_path = f'checkpoint/{identifier}.pth'
        log_filename = f'log/{identifier}.csv'

        trainer = setup_training(model, train_loader, val_loader, args.num_epochs, checkpoint_path, log_filename, device)
        trainer.train()

        print(f'Training completed for model: {model_type}')
        print(f'Testing for model: {model_type}')   

        inference = ModelTester(model, test_loader, device, checkpoint_path)
        inference.test(test_loader)

        print(f'------------------ \n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation model.')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size for data loader. Default is 8.')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs for training. Default is 10.')

    args = parser.parse_args()
    main(args)