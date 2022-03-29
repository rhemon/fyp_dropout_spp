
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score as gmean_score

def confusion_matrix_dict(y_predicted, y_target):
    """
    Compoute confusion matrix 

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the confusion matrix.
    """
    cm = confusion_matrix(y_target, y_predicted)
    return f"Confusion matrix = {cm}\n"
    
def get_f1_score(y_predicted, y_target):
    """
    Compoute f1 score 

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the f1 score.
    """
    return f"F1 Score = {f1_score(y_target, y_predicted, average='macro')}\n"
    
def get_precision(y_predicted, y_target):
    """
    Compoute precision. 

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the precision.
    """
    return f"Precision = {precision_score(y_target, y_predicted, average='macro')}\n"

def get_recall(y_predicted, y_target):
    """
    Compoute recall.

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the recall.
    """
    return f"Recall = {recall_score(y_target, y_predicted, average='macro')}\n"

def get_gmean(y_predicted, y_target):
    """
    Compoute g-mean 

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the g-mean.
    """
    return f"Geometric Mean = {gmean_score(y_target, y_predicted)}\n"

def get_accuracy(y_predicted, y_target):
    """
    Compoute accuracy.

    @param y_predicted : Predicted class values.
    @param y_target    : Target class values.

    @return String describing the accuracy.
    """
    return f"Accuracy = {accuracy_score(y_target, y_predicted)}\n"

def get_evaluation_methods(cfg):
    """
    Get list of evaluation functions specified in configuration.

    @param cfg : SimpleNamespace object of configuration read from json file.

    @return List of functions that take y_predicted and y_target as input and
            return a string containing the metric name and score of it.
    """
    evaluation_methods = cfg.EVALUATION_METHODS
    methods = []
    for each in evaluation_methods:
        if each == "CONFUSION_MATRIX":
            methods.append(confusion_matrix_dict)
        elif each == "ACCURACY":
            methods.append(get_accuracy)
        elif each == "F1_SCORE":
            methods.append(get_f1_score)
        elif each == "PRECISION":
            methods.append(get_precision)
        elif each == "RECALL" or each == "UAR":
            methods.append(get_recall)
        elif each == "GMEAN":
            methods.append(get_gmean)
        else:
            raise Exception(f"Requested evaluation method {each} is not supported")
    return methods 
