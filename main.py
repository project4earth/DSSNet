import warnings
import torch
import argparse
import torch.optim as optim
from datetime import datetime
from model.a2fpn import A2FPN
from model.cmlformer import CMLFormer
from model.cmtfnet import CMTFNet
from model.dcswin import DCSwin
from model.deeplabv3plus import DeepLabv3Plus
from model.hrnet import HRNet
from model.segformer import SegFormer
from model.unet import UNet
from model.unetformer import UNetFormer
from train import ModelTrainer
from test import ModelTester
from dataset import get_dataloaders
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

def initialize_model(model_type, in_channels, num_classes):
    model_dict = {
        'deeplabv3plus': DeepLabv3Plus(in_channels=in_channels, num_classes=num_classes),
        'segformer': SegFormer(in_channels=in_channels, num_classes=num_classes), 
        'hrnet': HRNet(in_channels=in_channels, num_classes=num_classes),
        'unet': UNet(in_channels=in_channels, num_classes=num_classes),
        'unetformer': UNetFormer(in_channels=in_channels, num_classes=num_classes), 
        'cmlformer': CMLFormer(in_channels=in_channels, num_classes=num_classes),
        'cmtfnet': CMTFNet(in_channels=in_channels, num_classes=num_classes),
        'dcswin': DCSwin(in_channels=in_channels, num_classes=num_classes),
        'a2fpn': A2FPN(in_channels=in_channels, num_classes=num_classes),
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
    model_types = ['a2fpn', 'unetformer', 'cmlformer', 'cmtfnet', 'dcswin', 'a2fpn', 'deeplabv3plus', 'segformer', 'hrnet']

    for model_type in model_types:
        print(f'Using model: {model_type}')

        in_chn = config.NUM_CHANNELS_1 + config.NUM_CHANNELS_2
        model = initialize_model(model_type=model_type, in_channels=in_chn, num_classes=config.NUM_CLASSES)
        model = model.to(device)

        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = f'{model_type}_stacked_{config.NUM_CHANNELS}in_lr{config.LEARNING_RATE}_{config.NUM_EPOCHS}epochs_{current_datetime}'
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