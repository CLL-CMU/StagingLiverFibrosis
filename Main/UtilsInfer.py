import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from .Loss.ordinal_regression import prediction2label
from .UtilsTrain import initialize_model, create_datasets_and_loaders


@torch.no_grad()
def infer_casesdata(model, loader, device):
    """
    Perform inference on the given data loader and return predictions, classes, and labels.
    
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        tuple: Tuple containing predictions, predicted classes, and true labels.
    """
    model.eval()
    accu_num = torch.zeros(1, device=device)
    sample_num = 0
    all_preds, all_labels, all_cls = [], [], []
    
    loader = tqdm(loader, file=sys.stdout, desc="[Inference]")  # Display progress bar
    
    for step, batch_data in enumerate(loader):
        image1s, labels = [x.to(device).float() for x in batch_data]
        labels = labels.to(device).long()
        sample_num += image1s.size(0)
        outputs = model(image1s)
        probabilities = outputs
        pred_classes = prediction2label(outputs)
        accu_num += torch.sum(pred_classes == labels).item()
        all_preds.append(probabilities.cpu().numpy())
        all_cls.append(pred_classes.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        avg_acc = accu_num.item() / sample_num if sample_num != 0 else 0
        loader.set_postfix(acc=avg_acc)  # Show current average accuracy in the progress bar
    loader.close()
    all_preds = np.concatenate(all_preds)
    all_cls = np.concatenate(all_cls)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_cls, all_labels

def collect_datainfo(args, device):
    """
    Initialize the model and perform inference on specified datasets (train, valid, test).
    
    Args:
        args (argparse.Namespace): Configuration parameters.
        device (torch.device): Device to run the model on (CPU or GPU).
        
    Returns:
        dict: Dictionary of tuples containing metrics for each dataset type.
    """
    model, _, _, _ = initialize_model(args, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    # Dictionary to hold metric results
    metric_results = {
        'train': infer_casesdata(model, create_datasets_and_loaders(args, 'train', False), device),
        'valid': infer_casesdata(model, create_datasets_and_loaders(args, 'valid', False), device),
        'internal_test': infer_casesdata(model, create_datasets_and_loaders(args, 'internal_test', False), device),
        'external_test': infer_casesdata(model, create_datasets_and_loaders(args, 'external_test', False), device)
    }
    return metric_results

def write_combined_csv(metric_results,logging, args):
    """
    Write combined inference results across all datasets (train, valid, test) to a CSV file.
    
    Args:
        metric_results (dict): Dictionary containing metric info for each dataset type.
        args (argparse.Namespace): Configuration parameters.
    """
    combined_data = []
    
    anno_file_map = {
        'train': args.train_anno_file,
        'valid': args.val_anno_file,
        'internal_test': args.internal_test_anno_file,
        'external_test': args.external_test_anno_file
    }
    
    for dataset_type, metric_info in metric_results.items():
        score_info, cls_info, lab_info = metric_info
        anno_file = anno_file_map[dataset_type]
        anno_info = np.loadtxt(anno_file, dtype=str, delimiter=',', usecols=(0,))
        
        for id_, true_label, pred_label, score in zip(anno_info, lab_info, cls_info, score_info):
            combined_data.append({
                'ID': id_,
                'True_Label': true_label,
                'Predicted_Label': pred_label,
                'Score_Class_0': score[0],
                'Score_Class_1': score[1],
                'Score_Class_2': score[2],
                'Score_Class_3': score[3],
                'Dataset_Type': dataset_type
            })
    
    df = pd.DataFrame(combined_data)
    csv_file_path = os.path.join(args.results_dir, f"{args.mode}_fold{args.fold}_combined_results.csv")
    df.to_csv(csv_file_path, index=False)
    return csv_file_path
    

