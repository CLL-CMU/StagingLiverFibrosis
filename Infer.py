import os
import torch
import time
from Main.UtilsInfer import collect_datainfo, write_combined_csv
import logging
import sys

# Set up logging
def setup_logging(log_file_path):
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path)])
    logger = logging.getLogger(__name__)
    return logger


def InferMain(args,logging):
    """
    Main entry point for the script.
    
    Args:
        args (argparse.Namespace): Configuration parameters.
    """
    logging.info("Inference started...")
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric_results = collect_datainfo(args, device)
    result_path = write_combined_csv(metric_results,logging, args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Inference completed in {elapsed_time:.2f} seconds.")
    logging.info(f"Predictions have been saved to '{result_path}'.")


if __name__ == "__main__":
    import argparse

    def parse_arguments():
        """
        Parse command-line arguments for inference.
        """
        parser = argparse.ArgumentParser(description='Infer classification models for MRI images')
        parser.add_argument('--model_path', type=str, default="best_acc_model", help='(e.g., best_acc_model, best_loss_model)')
        parser.add_argument('--num_classes', type=int, default=4, help='Number of classes for classification')
        parser.add_argument('--target_shapes', type=tuple, default=(36, 168, 192), help='Target shapes')
        parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
        parser.add_argument('--gpus', default='0,1', type=str, help='GPU device IDs to use (e.g., "0,1,2")')
        parser.add_argument('--data_dir', type=str, required=True, help='Directory with data')
        parser.add_argument('--mode', type=str, default='T2FS', help='MRI image type (e.g., T1, T2FS)')
        parser.add_argument('--fold', type=int, default=1, help='Fold number for cross-validation')
        parser.add_argument('--results_dir', type=str,  default="Results", help='Directory to save inference results')
        parser.add_argument('--anno_root', type=str, default="Txtfile", help='Root directory of annotation files')
        parser.add_argument('--train_anno_file', type=str, default="train_fold{fold}.txt", help='Path to training annotation file')
        parser.add_argument('--val_anno_file', type=str, default="val_fold{fold}.txt", help='Path to validation annotation file')
        parser.add_argument('--internal_test_anno_file', type=str, default="internal_test.txt", help='Path to test annotations file')
        parser.add_argument('--external_test_anno_file', type=str, default="external_test.txt", help='Path to test annotations file')
        parser.add_argument('--log_file', type=str, default="inference", help='Path to save the inference log file')
        args = parser.parse_args()
        # Update file paths based on the fold parameter
        fold_str = f"{args.fold}"
        args.train_anno_file = os.path.join(args.anno_root, args.train_anno_file.format(fold=fold_str))
        args.val_anno_file = os.path.join(args.anno_root, args.val_anno_file.format(fold=fold_str))
        args.internal_test_anno_file = os.path.join(args.anno_root, args.internal_test_anno_file)
        args.external_test_anno_file = os.path.join(args.anno_root, args.external_test_anno_file)

        args.model_path = os.path.join(args.results_dir, args.mode, f"fold{fold_str}", args.model_path + ".pth")
        args.results_dir = os.path.join(args.results_dir, args.mode)
        args.log_file = os.path.join(args.results_dir, args.log_file + "_fold" + fold_str + ".log")
        return args

    args = parse_arguments()
    logger = setup_logging(args.log_file)
    InferMain(args, logger)

