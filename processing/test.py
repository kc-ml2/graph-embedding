import torch
from utils import utils

"""
    Test the Model
"""
def test(model, test_data, device, logger) -> None:
    # merge test data & model data
    output = []
    label = []
    for data in test_data:
        data.to(device)
        log_p = model(data.x, data.edge_index)
        output.append(log_p)
        label.append(data.y)

    # prediction
    result = torch.cat(output, dim = 0)
    result = torch.sigmoid(result)
    # answer
    answer = torch.cat(label, dim = 0)
    
    # calculate accuracy
    acc = utils.accuracy(result, answer)
    logger.log("Final Accuracy is = {}".format(acc), "TEST")