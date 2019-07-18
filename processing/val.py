# For validation
import torch
import torch.nn.functional as F
from utils import utils

# For logging
import traceback

"""
    Validate the Model
"""

def val(epoch, batch, model, loss_ft, logger, is_full_data = False) -> None:
    model.eval()
    # put into model
    log_prob = model(batch.x, batch.edge_index)

    # split data if data is not splitted into train and val
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        # Select loss ft
        if loss_ft == "nll":
            loss_val = F.nll_loss(log_prob, label.long())
        elif loss_ft == "cross_entropy":
            loss_val = F.cross_entropy(log_prob, label.long())
        elif loss_ft == "mse":
            loss_val = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            # bce loss requires additional parsing to 0~1
            log_prob = torch.exp(log_prob)
            loss_val = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not Implemeted')
        # Calculate accuracy
        acc = utils.accuracy(log_prob, label)
        # pdb.set_trace()
        logger.log('epoch : {} || loss_val : {:.4f} || Accuracy: {:.4f}'.format(epoch, loss_val, acc), "VAL")
    except Exception:
        logger.log(traceback.format_exc(), "ERROR")