import torch

def save_model(model, optimizer, epoch, val_loss, other_params, save_path):
    # Save the state of the model, optimizer, and other parameters
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
    other_params = checkpoint.get('other_params', {})  # Retrieve other parameters, default to an empty dictionary if not found

    return model, optimizer, epoch, val_loss, other_params
