# for train
import torch
import torch.nn.functional as F

# for logging
import traceback

"""
    Train the model
"""

def train(epoch, batch, model, loss_ft, optimizer, logger, is_full_data = False) -> None:
    model.train()
    optimizer.zero_grad()

    # run model
    log_prob = model(batch.x, batch.edge_index)
    
    # Split the data if data is not splitted into train and val
    if is_full_data:
        log_prob = log_prob[batch.train_mask]
        label = batch.y[batch.train_mask]
    else:
        label = batch.y
    try:
        # Select Loss ft
        if loss_ft == 'nll':
            loss = F.nll_loss(log_prob, label.long())
        elif loss_ft == 'cross_entropy':
            loss = F.cross_entropy(log_prob, label.long())
        elif loss_ft == 'mse':
            loss = F.mse_loss(log_prob, label)
        elif loss_ft == 'bce':
            # bce loss have to be parsed to 0~1
            log_prob = torch.exp(log_prob)
            loss = F.binary_cross_entropy(log_prob, label)
        else:
            raise NotImplementedError('Not implemented')

        # Back propagate and learn
        loss.backward()
        optimizer.step()

        #print losses
        logger.log("Epoch {} | Loss {:.4f}".format(\
                epoch, loss.item()), "TRAIN")
    
    except Exception:
        logger.log(traceback.format_exc(), "ERROR")