def accuracy(log_prob, labels):
    pred = (log_prob>0.5).float()
    correct = (pred == labels).float().sum()
    return correct / len(labels)
