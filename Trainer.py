import argparse
import json
import logging
import os
import torch
from Main.UtilsTrain import initialize_model, create_datasets_and_loaders, train

def parse_arguments():
    """
    Parse command-line arguments for training classification models for fibrosis staging using non-contrast MRI (T1WI or T2FS).
    """
    parser = argparse.ArgumentParser(description='Train a neural network model for MRI classification')

    # General training configuration
    parser.add_argument('--mode', type=str, default='T2FS', help='MRI image type (e.g., T1, T2FS)') 
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for classification')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--target_shapes', type=tuple, default=(36, 168, 192), help='Target shapes')
    parser.add_argument('--angle', default=45, type=int, help='Rotation angle')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='Probability of random flip')
    parser.add_argument('--intensity_prob', type=float, default=0.25, help='Random intensity probability')
    parser.add_argument('--intensity_factor', type=float, default=0.1, help='Random intensity factor')
    parser.add_argument('--train_transform_list', type=str, default='["z_flip", "x_flip", "y_flip", "rotation", "random_intensity"]', help='List of training transformations')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with data')
    parser.add_argument('--anno_root', type=str, default="Txtfile", help='Root directory of annotation files')
    parser.add_argument('--results_root', type=str, default="Results", help='Root directory for TensorBoard logs and model checkpoints')
    parser.add_argument('--gpus', default='0', type=str, help='GPU device IDs to use (e.g., "0,1,2")')

    # Cross-validation fold configuration
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Current fold number (1-5)')
    parser.add_argument('--train_anno_file', type=str, default="train_fold{fold}.txt", help='Path to training annotation file')
    parser.add_argument('--val_anno_file', type=str, default="val_fold{fold}.txt", help='Path to validation annotation file')
    parser.add_argument('--tensorboard_dir', type=str, default="fold{fold}/", help='TensorBoard log directory')
    parser.add_argument('--checkpoint_path', type=str, default="fold{fold}/model_checkpoint.pth", help='Model checkpoint path')
    parser.add_argument('--resume_training', action="store_true", help='Whether to resume training from a checkpoint')
    parser.add_argument('--log_dir', type=str, default="fold{fold}/training.log", help='Path to save the training log')

    args = parser.parse_args()

    # Update file paths based on the fold parameter
    fold_str = f"{args.fold}"
    mode_str = args.mode
    args.train_anno_file = os.path.join(args.anno_root, args.train_anno_file.format(fold=fold_str))
    args.val_anno_file = os.path.join(args.anno_root, args.val_anno_file.format(fold=fold_str))
    args.tensorboard_dir = os.path.join(args.results_root, mode_str, args.tensorboard_dir.format(fold=fold_str))
    args.checkpoint_path = os.path.join(args.results_root, mode_str, args.checkpoint_path.format(fold=fold_str))
    args.log_dir = os.path.join(args.results_root, mode_str, args.log_dir.format(fold=fold_str))

    # Save configuration to JSON
    json_file_path = os.path.join(args.tensorboard_dir, f"{mode_str}_fold{fold_str}_config.json")
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    with open(json_file_path, 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

    return args

def setup_logging(log_dir):
    """
    Set up logging to a file and to the console.
    
    Args:
        log_dir (str): The directory where log files will be saved.
    """
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    # Set the logging level and format
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_dir),
                            logging.StreamHandler()
                        ])

def main(args):
    """
    Main entry point of the script.
    """
    setup_logging(args.log_dir)
    logging.info("Training started...")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, criterion, and scheduler
    model, optimizer, criterion, scheduler = initialize_model(args, device)

    # Create data loaders for training, validation, and test sets
    loaders_train = create_datasets_and_loaders(args, dataset_type='train', is_training=True)
    loaders_val = create_datasets_and_loaders(args, dataset_type='valid', is_training=False)

    # Train the model
    train(model, loaders_train, loaders_val, optimizer, criterion, scheduler, device, logging, args)

    # Log completion message
    logging.info("Training completed!")

if __name__ == '__main__':
    options = parse_arguments()
    main(options)
