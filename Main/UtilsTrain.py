import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from .Loss.ordinal_regression import ordinal_regression,prediction2label
from .Datasets.MRIDataset import MRIDataset, create_loader
from .Model.CoTNet3D_Model import CoTNet3D_Model
from .Model.Load_save_models import save_model,load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

def initialize_model(args, device):
    """
    Initialize the model, loss function, optimizer, and scheduler.
    
    Args:
        args (argparse.Namespace): The arguments containing configuration.
        device (torch.device): The device to run the model on (CPU or GPU).
        
    Returns:
        tuple: Contains the model, optimizer, loss function, and scheduler.
    """
    # Create an instance of the CoTNet3D model with the specified number of classes
    model = CoTNet3D_Model(num_classes=args.num_classes)
    
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Wrap the model to run on multiple GPUs
        model = torch.nn.DataParallel(model)
        
    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)
    
    # Print the model architecture
    print(model)
    
    # Initialize the optimizer (Adam in this case) with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Specify the loss function (Ordinal Regression in this case)
    criterion = ordinal_regression
    
    # Initialize the learning rate scheduler (ReduceLROnPlateau in this case) to adjust the learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    return model, optimizer, criterion, scheduler


def create_datasets_and_loaders(args, dataset_type, is_training):
    """
    Create datasets and data loaders for training or evaluation.
    
    Args:
        args (argparse.Namespace): The arguments containing configuration.
        dataset_type (str): Type of the dataset ('train', 'valid', 'test').
        is_training (bool): Flag indicating if the loader is for training.
        
    Returns:
        DataLoader: The data loader for the specified dataset type.
    """
    # Initialize the dataset with the provided arguments, dataset type, and training flag
    dataset = MRIDataset(args, dataset_type=dataset_type, is_training=is_training)
    
    # Create the data loader from the dataset with the specified batch size and training flag
    loader = create_loader(dataset, batch_size=args.batch_size, is_training=is_training)
    
    return loader



def train_one_epoch(args, model, loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        args (argparse.Namespace): The arguments containing configuration.
        model (torch.nn.Module): The model to train.
        loader (DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (callable): The loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
        epoch (int): The current epoch number.
        
    Returns:
        tuple: The average loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    accu_loss = torch.zeros(1, device=device)  # Initialize the accumulated loss counter
    accu_num = torch.zeros(1, device=device)   # Initialize the accumulated correct predictions counter
    sample_num = 0  # Initialize the counter for total samples processed
    loader = tqdm(loader, file=sys.stdout, desc="[Training Epoch {}]".format(epoch))  # Use tqdm to display the training progress
    
    for step, batch_data in enumerate(loader):  # Iterate through each batch in the data loader
        # Move the images and labels to the specified device and convert to float
        image1s, labels = [x.to(device).float() for x in batch_data]
        labels = labels.to(device).long()  # Convert labels to long type
        sample_num += image1s.size(0)  # Update the total number of samples processed

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(image1s)  # Perform forward pass through the model
        pred_classes = prediction2label(outputs)  # Convert model outputs to predicted classes
        accu_num += torch.sum(pred_classes == labels).item()  # Update the accumulated correct predictions counter

        loss = criterion(outputs, labels)  # Calculate the loss
        accu_loss += loss.item()  # Accumulate the loss
        
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        
        # Update tqdm progress display with current loss and accuracy
        loader.set_postfix(loss=accu_loss.item() / (step + 1), acc=accu_num.item() / sample_num)

    loader.close()  # Close the tqdm progress display
    
    # Return the average loss and accuracy for the epoch
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num




@torch.no_grad()  # Disable gradient calculation to save compute resources and memory
def evaluation(args, model, loader, criterion, device, epoch, val_test='valid'):
    """
    Evaluate the model on the validation or test set.
    
    Args:
        args (argparse.Namespace): The arguments containing configuration.
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The data loader for the validation/test data.
        criterion (callable): The loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
        epoch (int): The current epoch number.
        val_test (str): Flag indicating whether it is validation ('valid') or test ('test') evaluation.
        
    Returns:
        tuple: The average loss and accuracy for the evaluation.
    """
    model.eval()  # Set the model to evaluation mode
    accu_num = torch.zeros(1, device=device)  # Initialize the accumulated correct predictions counter
    accu_loss = torch.zeros(1, device=device)  # Initialize the accumulated loss counter
    sample_num = 0  # Initialize the total samples processed counter
    all_labels = []  # List to store all true labels
    
    # Use tqdm to display evaluation progress
    loader = tqdm(loader, file=sys.stdout, desc="[Evaluating Epoch {}]".format(epoch))
    
    for step, batch_data in enumerate(loader):  # Iterate through each batch in the data loader
        # Move images and labels to the specified device and convert to float
        image1s, labels = [x.to(device).float() for x in batch_data]
        labels = labels.to(device).long()  # Convert labels to long type
        sample_num += image1s.size(0)  # Update the total number of samples processed
        outputs = model(image1s)  # Perform forward pass through the model
        
        loss = criterion(outputs, labels)  # Calculate the loss
        accu_loss += loss.item()  # Accumulate the loss
        
        pred_classes = prediction2label(outputs)  # Convert model outputs to predicted classes
        accu_num += torch.sum(pred_classes == labels).item()  # Update the accumulated correct predictions counter
        
        # Collect true labels in a list for further evaluation metrics if needed
        all_labels.extend(labels.cpu().numpy())
        
        # Calculate the average loss and accuracy
        avg_loss = accu_loss.item() / (step + 1) if step != 0 else 0
        avg_acc = accu_num.item() / sample_num if sample_num != 0 else 0
        
        # Update tqdm progress display with current loss and accuracy
        loader.set_postfix(loss=avg_loss, acc=avg_acc)
        
    loader.close()  # Close the tqdm progress display
    
    # Return the average loss and accuracy for the evaluation
    return avg_loss, avg_acc



def train(model, train_loaders, val_loaders, optimizer, criterion, scheduler, device, logging, args):
    """
    Train the model with the given data loaders, optimizer, criterion, and scheduler.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loaders (DataLoader): DataLoader for the training set.
        val_loaders (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (callable): The loss function.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to run the model on (CPU or GPU).
        args (argparse.Namespace): The arguments containing configuration.
        
    Returns:
        None
    """
    # Log the number of samples in the training and validation sets
    logging.info(f"Training on {len(train_loaders.dataset)} samples, validating on {len(val_loaders.dataset)} samples")
    
    # Create TensorBoard log directory
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    # Create an instance of SummaryWriter for TensorBoard logging
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    start_epoch = 0
    best_val_acc = -np.inf  # Initialize best validation accuracy
    best_val_loss = float('inf')  # Initialize best validation loss

    # If resuming training and the checkpoint file exists
    if args.resume_training and os.path.exists(args.checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {args.checkpoint_path}")
        # Load the previous model and optimizer state
        model, optimizer, start_epoch, best_val_loss, other_params = load_checkpoint(model, optimizer, args.checkpoint_path)
        # Update the scheduler's learning rate
        for _ in range(start_epoch):
            scheduler.step(best_val_loss)
    else:
        other_params = {}

    # Main training loop
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        # Train for one epoch and return training loss and accuracy
        train_loss, train_acc = train_one_epoch(args, model, train_loaders, optimizer, criterion, device, epoch)
        # Evaluate model on validation set and return loss, and accuracy
        val_loss, val_acc = evaluation(args, model, val_loaders, criterion, device, epoch, val_test='valid')
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        end_time = time.time()
        # Calculate the duration of the epoch
        epoch_duration = end_time - start_time
        
        # Save model checkpoint
        save_model(model, optimizer, epoch, val_loss, other_params, args.checkpoint_path)
        
        # Write training and validation metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        
        # Log training results for the epoch
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs} => "
                     f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f};  "
                     f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f};  "
                     f"Epoch Time: {epoch_duration:.2f} seconds.")
        
        # Save the model if the current validation accuracy is higher than the best seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"best_acc_model.pth"
            os.makedirs(os.path.join(args.tensorboard_dir), exist_ok=True)
            model_filepath = os.path.join(args.tensorboard_dir, model_filename)
            torch.save(model.state_dict(), model_filepath)
        
        # Save the model if the current validation loss is lower than the best seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filename = f"best_loss_model.pth"
            os.makedirs(os.path.join(args.tensorboard_dir), exist_ok=True)
            model_filepath = os.path.join(args.tensorboard_dir, model_filename)
            torch.save(model.state_dict(), model_filepath)
        
        # Log the model save path and best validation performance
        logging.info(f"Model saved to {model_filepath}")
        logging.info(f"Best performance => Val Acc: {best_val_acc:.4f}")
    
    # Close the TensorBoard's SummaryWriter at the end of training
    writer.close()
