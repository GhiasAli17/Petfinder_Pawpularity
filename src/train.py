import torch
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as sklearn_rmse
from utils.helpers import MetricMonitor


def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None):
    """
    Performs one epoch of training, handles mixup, and calculates RMSE.
    """
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    all_train_preds = []
    all_train_targets = []

    for i, (images, dense, target) in enumerate(stream, start=1):
       
        images = images.to(params['device'], non_blocking=True)
        dense = dense.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True).float().view(-1, 1)
        
        output = model(images, dense)
        loss = criterion(output, target)
        
        all_train_preds.extend(torch.sigmoid(output).detach().cpu().numpy().flatten() * 100)
        all_train_targets.extend(target.detach().cpu().numpy().flatten() * 100)


        metric_monitor.update('Loss', loss.item())
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

    epoch_rmse = sklearn_rmse(all_train_targets, all_train_preds)
    print(f"Epoch {epoch:02} Train RMSE: {epoch_rmse:.3f}")
    

    return epoch_rmse