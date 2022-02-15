
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import geometric_mean_score as gmean_score

def confusion_matrix_dict(y_predicted, y_target):
    tn, fp, fn, tp = confusion_matrix(y_target, y_predicted)
    return f"TN = {tn}\nTP = {tp}\nFN = {fn}\nFP = {fp}\n"
    

def get_f1_score(y_predicted, y_target):
    return f"F1 Score = {f1_score(y_target, y_predicted)}\n"
    
def get_precision(y_predicted, y_target):
    return f"Precision = {precision_score(y_target, y_predicted)}\n"

def get_recall(y_predicted, y_target): ### TODO: UAR consideration
    return f"Recall = {recall_score(y_target, y_predicted)}\n"

def get_roc_auc(y_predicted, y_target):
    return f"ROC AUC = {roc_auc_score(y_target, y_predicted)}\n"

def get_gmean(y_predicted, y_target):
    return f"Geometric Mean = {gmean_score(y_target, y_predicted)}\n"


def get_evaluation_methods(cfg):
    evaluation_methods = cfg.EVALUATION_METHODS
    methods = []
    for each in evaluation_methods:
        if each == "CONFUSION_MATRIX":
            methods.append(confusion_matrix_dict)
        elif each == "F1_SCORE":
            methods.append(get_f1_score)
        elif each == "PRECISION":
            methods.append(get_precision)
        elif each == "RECALL" or each == "UAR":
            methods.appendd(get_recall)
        elif each == "GMEAN":
            methods.append(get_gmean)
        else:
            raise Exception(f"Requested evaluation method {each} is not supported")
    return methods 