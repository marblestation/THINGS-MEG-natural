#!/usr/bin/env python3

import os
import sys
import random
import argparse
import mne
import numpy as np
import pandas as pd
from mne.decoding import LinearModel, SlidingEstimator
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import tools

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Choose a machine learning model.")
    valid_models = [
        "LinearDiscriminantAnalysis",
        "StandardScalerLinearDiscriminantAnalysis",
        "LogisticRegression",
        "GaussianNB",
        "RandomForestClassifier",
        "MLPClassifier",
        "XGBClassifier",
    ]
    parser.add_argument("model", type=str, choices=valid_models, help="Specify the model to use.")
    args = parser.parse_args()
    model = args.model

    # For reproducibility
    base_random_seed = 42
    np.random.seed(base_random_seed)
    random.seed(base_random_seed)

    target_categories = ['natural', 'human_made']
    if len(target_categories) == 2:
        # If it is a binary classification, only one class needs to be model
        model_categories = [target_categories[0]]
    else:
        model_categories = target_categories

    homogenize = False
    categories = tools.data.read_categories()

    for participant in range(1, 4+1):
        for block in ('BASE', 'EARLY', 'LATE'):
            participants = { participant: tools.data.get_block_sessions(participant, block)}

            # Set a unique seed for each participant and session combination
            random_seed = base_random_seed + (participant * 100) + (session * 1000)
            np.random.seed(random_seed)

            pickle_filename = f'output/models/fitted_{model}_{block}_P{participant}_S{session:02}.pkl'
            os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
            if os.path.exists(pickle_filename):
                print(f"Output file already exists: '{pickle_filename}")
                continue

            # Dictionary to store fitted models and test scores per category.
            # For each category, we will save a list of dictionaries, each containing:
            #   - 'fold': the fold number,
            #   - 'estimator': the fitted SlidingEstimator,
            #   - 'score': the test score on that fold.
            fitted_models = {cat: {'folds': [], 'stats': {'bayes_factors': {}, 'p-values': {}}} for cat in model_categories}

            data, ch_names, ch_types, sampling_rate, description, events, event_id = tools.data.read_meg(participants, categories, target_categories, homogenize=homogenize)
            n_times = data.shape[2]
            tmin = description['tmin']
            tmax = description['tmax']
            times = np.linspace(tmin, tmax, n_times)

            events_df = tools.data.enrich(events, categories)
            data, events, events_df = tools.data.filter(data, events, events_df)
            if homogenize:
                data, events, events_df = tools.data.homogenize(data, events, events_df, target_categories)

            # multi-label target matrix
            y_multi = events_df[target_categories].values  # shape: (n_trials, n_categories)

            #--------------------------------------------------------------------------------
            # MODEL
            #--------------------------------------------------------------------------------
            # MEG data is high-dimensional with many correlated channels:
            # - Methods like LDA, XGBClassifier, and RandomForestClassifier can exploit the covariance structure
            # - Logistic Regression works with correlated channels but does not model covariance
            # - Methods such as GNB (which assumes feature independence) may underperform if channels are strongly correlated.
            # - KNeighbors does not work well with high-dimensional data
            #
            # Dataset size:
            # - Small dataset: Simpler, low-parameter models (e.g., LDA, Logistic Regression, GNB) are often preferable
            # - Large dataset: More complex/flexible methods (e.g., MLPClassifier, XGBClassifier) can be considered. KNeighbors and SVC scale poorly.
            #
            # Computation:
            # - Fast: Linear models (LDA, Logistic Regression, or linear SVC with small dataset) are faster but may miss non-linear features
            # - Slow: Tree-based and neural network methods capture nonlinearities and interactions more naturally at the cost of increased computational demand and risk of overfitting
            #
            # LinearModel:
            # - It wrap a linear estimator so that its learned coefficients can be transformed into interpretable activation patterns
            #    - raw classifier weights (or filters) are not directly interpretable as brain activation patterns due to the mixing of signal and noise
            #    - it applies the “Haufe transformation” to convert these weights into patterns that better reflect the underlying neural sources
            #        coef = get_coef(time_decod, "patterns_", inverse_transform=True)
            #        evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
            #        joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
            #        evoked_time_gen.plot_joint(
            #            times=np.arange(0.0, 0.500, 0.100), title="patterns", **joint_kwargs
            #        )
            # - It is meant for linear classifiers where the decision function is a linear combination of the inputs (LDA, Logistic Regression)
            #--------------------------------------------------------------------------------
            #
            # | Situation                             | LDA  | GNB       | Logistic Regression            | XGBClassifier       | MLPClassifier                 | RandomForestClassifier | KNeighborsClassifier | SVC                         |
            # |---------------------------------------|------|-----------|--------------------------------|---------------------|-------------------------------|------------------------|----------------------|-----------------------------|
            # | Features are correlated               | Yes  | No        | Yes (doesn’t model covariance) | Yes                 | Yes                           | Yes                    | Yes                  | Yes                         |
            # | Features are independent              | No   | Yes       | Yes                            | Yes                 | Yes                           | Yes                    | Yes                  | Yes                         |
            # | Large dataset                         | Yes  | Yes       | Yes                            | Yes                 | Yes                           | Yes                    | No (scales poorly)   | No (scales poorly)          |
            # | Small dataset                         | Yes  | No        | Yes                            | No                  | No                            | Yes                    | Yes                  | Yes                         |
            # | Decision boundary is linear           | Yes  | No        | Yes                            | Yes/No (tree depth) | No                            | No                     | No                   | Yes (for linear kernel)     |
            # | Decision boundary is nonlinear        | No   | Yes       | No                             | Yes                 | Yes                           | Yes                    | Yes                  | Yes (for RBF/poly kernels)  |
            # | Works well with high-dimensional data | No   | Yes       | Yes                            | Yes                 | Yes                           | Yes                    | No                   | No (can be slow)            |
            # | Handles interactions between features | Yes  | No        | No                             | Yes                 | Yes                           | Yes                    | Yes                  | Yes (for nonlinear kernels) |
            # | Robust to outliers                    | No   | No        | Yes (if using regularization)  | Yes                 | No (sensitive to weight init) | Yes                    | No                   | Yes (with certain kernels)  |
            # | Computational efficiency              | Fast | Very Fast | Fast                           | Slower              | Slowest                       | Slower                 | Slow                 | Slow                        |
            #--------------------------------------------------------------------------------
            # USABLE OPTIONS
            if model == "GaussianNB":
                base_clf = GaussianNB() # Naïve Bayes, 2m per fold
            elif model == "LinearDiscriminantAnalysis":
                #base_clf = LinearDiscriminantAnalysis() # 15m per fold
                base_clf = LinearModel(LinearDiscriminantAnalysis()) # 15m per fold
            elif model == "StandardScalerLinearDiscriminantAnalysis":
                #base_clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # 15m per fold
                base_clf = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis())) # 15m per fold
            elif model == "LogisticRegression":
                #base_clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight='balanced', random_state=random_seed)) # 15m per fold
                base_clf = make_pipeline(StandardScaler(), LinearModel((LogisticRegression(solver='liblinear', class_weight='balanced', random_state=random_seed)))) # 15m per fold
            elif model == "RandomForestClassifier":
                base_clf = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight="balanced", random_state=random_seed) # 25m per fold (not accurate enough)
            elif model == "MLPClassifier":
                #base_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=random_seed)) # 30m per fold
                base_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8,), max_iter=500, random_state=random_seed)) # 45m per fold
            elif model == "XGBClassifier":
                base_clf = XGBClassifier(eval_metric="logloss", random_state=random_seed) # 1h30 per fold
            else:
                raise Exception(f"Unknown model '{model}")
            #
            #--------------------------------------------------------------------------------
            # DISCARDED OPTIONS
            #-- Options that are too slow
            #base_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42) # 3h per fold (too slow, not completed)
            #base_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=True)) # extremely slow (not completed)
            #base_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight="balanced", probability=False)) # 41h per fold (not completed)
            #
            #-- Options that do not converge
            #base_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(10,), max_iter=50, random_state=42)) # 7m per fold (it does not converge, max_iter is reached)
            #base_clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)) # 12m per fold (it does not converge, max_iter is reached)
            #
            #-- Options that lead to inaccurate modeling
            #base_clf = RandomForestClassifier(n_estimators=25, class_weight="balanced", random_state=42) # 40m per fold (not accurate enough)
            #base_clf = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight="balanced", random_state=42) # 25m per fold (not accurate enough)
            #base_clf = KNeighborsClassifier(n_neighbors=5) # 5m per fold (big output, not accurate enough)
            #--------------------------------------------------------------------------------

            ## Set up the multi-label classifier using OneVsRest strategy
            # - OneVsRestClassifier provides a neat wrapper for converting a binary classifier into a multi-class classifier
            # - However, the problem has already been defined as a binary classifier, so using OneVsRestClassifier is redundant
            #multi_label_clf = OneVsRestClassifier(base_clf)
            multi_label_clf = base_clf

            k_folds = 5
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

            X = data
            for cat in model_categories:
                # Get the binary target for this category.
                cat_idx = target_categories.index(cat)
                y = y_multi[:, cat_idx]

                # Run k-folds
                fold_num = 0
                for train_idx, test_idx in cv.split(X, y):
                    # Clone the multi-label classifier so that each fold is independent.
                    # Note: SlidingEstimator wraps the classifier so that it fits one model per timepoint.
                    time_decod = SlidingEstimator(clone(multi_label_clf), scoring=None, n_jobs=4)
                    # Fit on training data (all timepoints are used; X has shape [n_trials, n_channels, n_times]).
                    time_decod.fit(X[train_idx], y[train_idx])

                    if len(target_categories) == 2:
                        # For each timepoint, extract predictions and probability estimates.
                        n_test = len(test_idx)
                        preds = np.zeros((n_test, n_times))
                        probas = np.zeros((n_test, n_times))
                        for t in range(n_times):
                            X_test_t = data[test_idx, :, t]
                            preds[:, t] = time_decod.estimators_[t].predict(X_test_t)
                            try:
                                # Get probability for class "1".
                                probas[:, t] = time_decod.estimators_[t].predict_proba(X_test_t)[:, 1]
                            except Exception as e:
                                print(type(e), e)
                                probas = None
                    else:
                        # Evaluate on the test data.
                        preds = time_decod.predict(data[test_idx])  # shape: (n_test_samples, n_times)
                        probas = None

                    metrics = tools.stats.compute_metrics(n_times, y[test_idx], preds, probas, target_categories)

                    # Store the fold number, the fitted estimator, and the test score.
                    fitted_models[cat]['folds'].append({
                        'fold': fold_num,
                        'model': model,
                        'block': block,
                        'estimator': time_decod,
                        'metrics': metrics,
                        'mean_score': np.mean(metrics['balanced_accuracy'], axis=0),
                        'sem_score': np.std(metrics['balanced_accuracy'], axis=0) / np.sqrt(len(metrics['balanced_accuracy'])),
                        'times': times,
                        'datapoints': {
                            'test': {
                                'target': len(np.where(y[test_idx] == 1)[0]),
                                'total': len(y[test_idx]),
                            },
                            'train': {
                                'target': len(np.where(y[train_idx] == 1)[0]),
                                'total': len(y[train_idx]),
                            },
                        }
                    })
                    print(f"Category {cat}, fold {fold_num}: test score = {fitted_models[cat]['folds'][-1]['mean_score']:.3f}")
                    fold_num += 1

                for m_name in ('balanced_accuracy', 'accuracy'):
                    collected_scores = np.asarray([fitted_models[cat]['folds'][i]['metrics'][m_name] for i in range(len(fitted_models[cat]['folds']))])
                    if "balanced" in m_name:
                        null_value = 0.5
                    else:
                        #null_value = 1 - (1 / len(target_categories))
                        null_value = len(np.where(y[test_idx] == 1)[0]) / len(y[test_idx])
                    fitted_models[cat]['stats']['bayes_factors'][m_name] = tools.stats.compute_bayes(n_times, collected_scores, null_value)
                    fitted_models[cat]['stats']['p-values'][m_name] = tools.stats.compute_p_values(n_times, collected_scores, null_value)

                if len(target_categories) == 2:
                    m_name = "mcc"
                    collected_scores = np.asarray([fitted_models[cat]['folds'][i]['metrics'][m_name] for i in range(len(fitted_models[cat]['folds']))])
                    null_value = 0.0
                    fitted_models[cat]['stats']['bayes_factors'][m_name] = tools.stats.compute_bayes(n_times, collected_scores, null_value)
                    fitted_models[cat]['stats']['p-values'][m_name] = tools.stats.compute_p_values(n_times, collected_scores, null_value)

            # Save
            joblib.dump(fitted_models, pickle_filename)
            print(f"Written '{pickle_filename}")

