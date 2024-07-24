import torch

def save_model(model, optimizer, epoch, val_loss, other_params, save_path):
    # 保存模型、优化器和其他参数的状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'other_params': other_params,
    }, save_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    other_params = checkpoint.get('other_params', {})  # 获取其他参数，如果不存在则默认为空字典
    
    return model, optimizer, epoch, val_loss, other_params
