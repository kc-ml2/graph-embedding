def accuracy(log_prob, labels):
    preds = log_prob.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
