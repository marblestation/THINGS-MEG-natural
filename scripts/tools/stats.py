import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, cohen_kappa_score)
import scipy.stats as stats
import pingouin as pg


#--------------------------------------------------------------------------------
# Helper functions for binary-specific metrics
def specificity(y_true, y_pred):
    """True Negative Rate: TN/(TN+FP)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN = cm[0, 0]
    FP = cm[0, 1]
    return TN / (TN + FP) if (TN + FP) > 0 else 0

def npv(y_true, y_pred):
    """Negative Predictive Value: TN/(TN+FN)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN = cm[0, 0]
    FN = cm[1, 0]
    return TN / (TN + FN) if (TN + FN) > 0 else 0
#--------------------------------------------------------------------------------

def compute_bayes(n_times, collected_scores, null_value):
    # Compute Bayes factors at each timepoint:
    bf = np.zeros(n_times)
    for t in range(n_times):
        # Use the cross-validation scores for this timepoint:
        scores_t = collected_scores[:, t]
        # Run a Bayesian one-sample t-test (alternative: accuracy > chance (0.5))
        res = pg.ttest(scores_t, null_value, alternative='greater', paired=False)
        bf[t] = res['BF10'].values[0]
    return bf

def compute_p_values(n_times, collected_scores, null_value):
    pvals = np.zeros(n_times)
    for t in range(n_times):
        scores_t = collected_scores[:, t]
        t_val, p_val = stats.ttest_1samp(scores_t, null_value)
        # For a one-tailed test (accuracy > chance), if t is positive, p_one = sf(t)
        if t_val > 0:
            pvals[t] = stats.t.sf(t_val, df=len(scores_t)-1)
        else:
            pvals[t] = 1.0
    return pvals

def get_metrics(target_categories):
    metrics_for_only_two = {
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef,
        'specificity': specificity,
        'sensitivity': recall_score,        # alias for recall
        'ppv': precision_score,             # alias for precision
        'npv': npv,
        'roc_auc': roc_auc_score,
    }

    metrics_for_two_or_more = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score,
    }

    metrics_dict = {}
    metrics_dict.update(metrics_for_two_or_more)
    if len(target_categories) == 2:
        metrics_dict.update(metrics_for_only_two)

    return metrics_dict

def compute_metrics(n_times, y_true, preds, probas, target_categories):
    metrics_dict = get_metrics(target_categories)
    metrics = {m: [] for m in metrics_dict.keys()}
    for t in range(n_times):
        y_pred = preds[:, t]
        for m_name, m_func in metrics_dict.items():
            if m_name == 'roc_auc' and probas is not None:
                metrics[m_name].append(float(m_func(y_true, probas[:, t])))
            else:
                metrics[m_name].append(float(m_func(y_true, y_pred)))
    return metrics


