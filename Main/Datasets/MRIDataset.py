import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import numpy as np
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
from .Transforms import *



class MRIDataset(Dataset):
    def __init__(self, args, dataset_type='train', is_training=True):
        """
        Dataset class for handling single-mode data.
        
        Args:
            args (Namespace): Arguments containing dataset information.
            dataset_type (str): Type of dataset ('train', 'val', 'test').
            is_training (bool): Flag indicating if the dataset is for training.
        """
        self.args = args
        self.is_training = is_training
        self.target_shapes = args.target_shapes
        
        img_list = []
        lab_list = []
        mode = args.mode
        
        # Load annotations based on dataset type
        if dataset_type == 'train':
            anno_file = args.train_anno_file
        elif dataset_type == 'valid':
            anno_file = args.val_anno_file
        elif dataset_type == 'internal_test': 
            anno_file = args.internal_test_anno_file
        elif dataset_type == 'external_test': 
            anno_file = args.external_test_anno_file

        annotations = np.loadtxt(anno_file, dtype=np.str_)
        for item in annotations:
            caseid = transform_string(item[0], mode)
            img_list.append(f'{args.data_dir}/{caseid}_0000.nii.gz')
            lab_list.append(int(item[1]))
        
        self.img_list = img_list
        self.lab_list = lab_list
    
    def __getitem__(self, index):
        """
        Get item by index.
        
        Args:
            index (int): Index of the item.
        
        Returns:
            tuple: Tuple containing image data and label.
        """
        img_path = self.img_list[index]
        img_data = read_one_mode_proimg(img_path, self.target_shapes)
        if self.is_training:
            img_data = self.apply_transforms(img_data, self.args.train_transform_list)
        label = self.lab_list[index]
        return img_data, label
    
    def apply_transforms(self, img_data, transform_list):
        """
        Apply a series of transformations to the image data.
        
        Args:
            img_data (np.ndarray): Image data.
            transform_list (list): List of transformations to apply.
        
        Returns:
            np.ndarray: Transformed image data.
        """
        args = self.args
        if 'z_flip' in transform_list:
            img_data = random_flip(img_data, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            img_data = random_flip(img_data, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            img_data = random_flip(img_data, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            img_data = rotate(img_data, args.angle)
        if 'random_intensity' in transform_list:
            img_data = random_intensity(img_data, args.intensity_factor, p=args.intensity_prob)
        return img_data
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.img_list)

def create_loader(dataset=None, batch_size=1, is_training=False, num_workers=16, distributed=False,
                  collate_fn=None, pin_memory=False, persistent_workers=True, worker_seeding='all'):
    """
    Create a data loader.

    Args:
        dataset (Dataset, optional): Dataset to load.
        batch_size (int): Number of samples per batch.
        is_training (bool): Whether the loader is for training.
        num_workers (int): Number of subprocesses to use for data loading.
        distributed (bool): Whether to use distributed sampler.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
        pin_memory (bool): Whether to pin memory.
        persistent_workers (bool): Whether to keep workers active.
        worker_seeding (str): Worker seed mode.

    Returns:
        DataLoader: PyTorch DataLoader
    """
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )

    loader = DataLoader(dataset, **loader_args)
    return loader

def main(args):
    """
    Main function to run the dataset and dataloader.

    Args:
        args (Namespace): Arguments containing configuration.
    """
    # Create dataset object
    dataset = MRIDataset(args, is_training=True)
    
    # Create data loader
    loader = create_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        is_training=True,
        # num_workers=args.num_workers,
        # distributed=args.distributed,
        # pin_memory=args.pin_memory,
    )
    
    # Iterate through the data loader
    for batch in loader:
        images, labels = batch
        print(f"Images shape: {images.shape}, Label: {labels}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-mode Dataset Loading')
    parser.add_argument('--mode', type=str, default='T1', help='Mode type')
    parser.add_argument('--target_shapes', type=int, nargs=3, default=(36, 168, 192), help='Target shapes')
    parser.add_argument('--train_anno_file', type=str, required=True, help='Path to training annotation file')
    parser.add_argument('--val_anno_file', type=str, required=True, help='Path to validation annotation file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with data')
    parser.add_argument('--flip_prob', type=float, default=0.5, help='Random flip probability')
    parser.add_argument('--intensity_prob', type=float, default=0.25, help='Random intensity probability')
    parser.add_argument('--intensity_factor', type=float, default=0.1, help='Random intensity factor')
    parser.add_argument('--angle', type=int, default=45, help='Maximum rotation angle')
    parser.add_argument('--train_transform_list', nargs='+', type=str, default=['z_flip', 'x_flip', 'y_flip', 'rotation', 'random_intensity'],
                        help='List of transformations to apply during training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--pin_memory', action='store_true', help='Use pinned memory')

    args = parser.parse_args()
    main(args)

