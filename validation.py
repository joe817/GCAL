from sklearn.metrics import roc_auc_score, f1_score

def validation(args, pred, y, graph, mask=None):
    if mask is not None:
        pred = pred[mask]
        y = y[mask]

    if args.dataset == "twitch":
        metric_result = roc_auc_score(y.cpu().numpy(), pred.cpu().numpy()) #roc_auc
    elif args.dataset == "elliptic":
        if mask is None:
            pred = pred[graph.mask]
            y = y[graph.mask]
        metric_result = f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro') #f1
    else:
        correct = (pred == y).sum().item()
        metric_result = correct / len(y) #acc
    return metric_result 

