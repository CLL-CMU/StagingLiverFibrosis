import os
import sys
import argparse
import torch
from Main.UtilsGradCAM import Infer_GradCAM_Main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="获取概率值")
    parser.add_argument('--gpus', default='1', type=str, help='GPU device IDs to use (e.g., "0,1,2")')
    parser.add_argument('--mode', type=str, default='T1', help='MRI图像类型')
    parser.add_argument('--num_classes', type=int, default=4, help='分类的类别数')
    parser.add_argument('--target_shapes', type=list, default=[36, 168, 192], help='目标形状')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--imagepath', type=str,default='/home/newroot/lcl/ai/code/MRI3Dfibrosis/Test/imagecrop/S1_T2FS_0000.nii.gz',  help='数据集根目录')
    parser.add_argument("--model_path", type=str, default='Results', help="模型路径")
    parser.add_argument("--model_weights", type=str, default='best_acc_model.pth', help="模型路径")
    parser.add_argument("--camoutput", type=str, default='/home/newroot/lcl/ai/code/MRIpaperCode/Results/11', help="模型路径")

    # 新增折数参数fold和Fstage
    parser.add_argument("--fold", type=int, default=2, choices=[1, 2, 3, 4, 5], help='当前的折数 (1-5)')

    # 解析参数
    args = parser.parse_args()
    
    # 根据fold参数更新文件路径
    fold_str = f"{args.fold}"
    # 构建和更新路径
    # args.model_path = os.path.join(args.model_path, args.mode, 'fold' + fold_str, args.model_weights)
    args.model_path = "/home/newroot/lcl/ai/code/MRIpaperCode/Results/T1/fold2/best_acc_model.pth"

    Infer_GradCAM_Main(args)
