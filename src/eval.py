import torch
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as sklearn_rmse
from utils.helpers import MetricMonitor

def validate_fn(val_loader, model, criterion, epoch, params):
    """
    Performs one epoch of validation and calculates RMSE.
    """
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for i, (images, dense, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            dense = dense.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True).float().view(-1, 1) 
            
            output = model(images, dense)
            loss = criterion(output, target)

            metric_monitor.update('Loss', loss.item())
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

            targets = (target.detach().cpu().numpy() * 100).tolist()
            outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)

    epoch_rmse = sklearn_rmse(final_targets, final_outputs)
    print(f"Epoch {epoch:02} Valid RMSE: {epoch_rmse:.3f}")
    

    return final_outputs, final_targets, epoch_rmse