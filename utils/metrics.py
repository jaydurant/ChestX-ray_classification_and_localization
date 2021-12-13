import numpy as np
from sklearn.metrics import *
    
def calculate_metrics(pred, target, threshold=0.5):

    threshpred = np.array(pred > threshold, dtype=float)

    return {'micro/precision': precision_score(y_true=target, y_pred=threshpred, average='micro'),

            'micro/recall': recall_score(y_true=target, y_pred=threshpred, average='micro'),

            'micro/f1': f1_score(y_true=target, y_pred=threshpred, average='micro'),

            'macro/precision': precision_score(y_true=target, y_pred=threshpred, average='macro'),

            'macro/recall': recall_score(y_true=target, y_pred=threshpred, average='macro'),
            
            'macro/f1': f1_score(y_true=target, y_pred=threshpred, average='macro'),

            'samples/precision': precision_score(y_true=target, y_pred=threshpred, average='samples'),

            'samples/recall': recall_score(y_true=target, y_pred=threshpred, average='samples'),

            'samples/f1': f1_score(y_true=target, y_pred=threshpred, average='samples'),

            #"weighted/auc": roc_auc_score(y_true=target, y_score=pred, average="weighted"),
            
            #"samples/auc": roc_auc_score(y_true=target, y_score=pred, average="samples")
}
