def accuracy(log_prob, labels):
    if log_prob.shape == labels.shape: 
        pred = (log_prob>0.5).float()
        print(pred[pred > 0])
        #print(pred)
        correct = (pred == labels).float().sum()
    else:
        _, pred = log_prob.max(dim=1)
        correct = float (pred.eq(labels).sum().item())
    return correct / len(labels)
