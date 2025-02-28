#!/usr/bin/env python3

import os
import sys
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
import pingouin as pg
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

    categories = tools.data.read_categories()
    target_categories = ['natural', 'human_made']
    if len(target_categories) == 2:
        # If it is a binary classification, only one class needs to be model
        model_categories = [target_categories[0]]
    else:
        model_categories = target_categories

    homogenize = False
    for participant in range(1, 4+1):
        block = "BASE"
        session = tools.data.get_block_sessions(participant, block)[0]
        pickle_filename = f'output/models/fitted_{model}_{block}_P{participant}_S{session:02}.pkl'
        if not os.path.exists(pickle_filename):
            print(f"Model file does not exists: '{pickle_filename}")
            continue
        fitted_models = joblib.load(pickle_filename)
        print(f"Read '{pickle_filename}")

        for block in ('EARLY', 'LATE'):
            # Set a unique seed for each participant and session combination
            random_seed = base_random_seed + (participant * 100) + (session * 1000)
            np.random.seed(random_seed)

            participants = { participant: tools.data.get_block_sessions(participant, block)}
            session = participants[participant][0]
            pickle_filename = f'output/models/predictions/predictions_{model}_{block}_P{participant}_S{session:02}.pkl'
            os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
            if os.path.exists(pickle_filename):
                continue
            data, ch_names, ch_types, sampling_rate, description, events, event_id = tools.data.read_meg(participants, categories, target_categories, homogenize=homogenize)

            events_df = tools.data.enrich(events, categories)
            data, events, events_df = tools.data.filter(data, events, events_df)
            if homogenize:
                data, events, events_df = tools.data.homogenize(data, events, events_df, target_categories)

            # multi-label target matrix
            y_multi = events_df[target_categories].values  # shape: (n_trials, n_categories)

            predictions = {cat: {'folds': [], 'stats': {'bayes_factors': {}, 'p-values': {}}} for cat in model_categories}

            k_folds = 5
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

            X = data
            n_times = X.shape[2]
            for cat in model_categories:
                # Get the binary target for this category.
                cat_idx = target_categories.index(cat)
                y = y_multi[:, cat_idx]

                # Run k-folds
                fold_num = 0
                for train_idx, test_idx in cv.split(X, y):
                    time_decod = fitted_models[cat]['folds'][fold_num]['estimator']

                    if len(target_categories) == 2:
                        # For each timepoint, extract predictions and probability estimates.
                        n_test = len(test_idx)
                        preds = np.zeros((n_test, n_times))
                        probas = np.zeros((n_test, n_times))
                        for t in range(n_times):
                            X_test_t = data[test_idx, :, t]
                            preds[:, t] = time_decod.estimators_[t].predict(X_test_t)
                            # Get probability for class "1".
                            probas[:, t] = time_decod.estimators_[t].predict_proba(X_test_t)[:, 1]
                    else:
                        # Evaluate on the test data.
                        preds = time_decod.predict(data[test_idx])  # shape: (n_test_samples, n_times)
                        probas = None

                    metrics = tools.stats.compute_metrics(n_times, y[test_idx], preds, probas, target_categories)

                    # Store the fold number, the fitted estimator, and the test score.
                    predictions[cat]['folds'].append({
                        'fold': fold_num,
                        'model': model,
                        #'block': fitted_models[cat]['folds'][fold_num]['block'],
                        #'estimator': time_decod,
                        'metrics': metrics,
                        'mean_score': np.mean(metrics['balanced_accuracy'], axis=0),
                        'sem_score': np.std(metrics['balanced_accuracy'], axis=0) / np.sqrt(len(metrics['balanced_accuracy'])),
                        'times': fitted_models[cat]['folds'][fold_num]['times'],
                        'datapoints': {
                            'validation': {
                                'target': len(np.where(y[test_idx] == 1)[0]),
                                'total': len(y[test_idx]),
                            },
                            #'test': fitted_models[cat]['folds'][fold_num]['datapoints']['test'],
                            #'train': fitted_models[cat]['folds'][fold_num]['datapoints']['train'],
                        }
                    })
                    print(f"Category {cat}, fold {fold_num}: test score = {predictions[cat]['folds'][-1]['mean_score']:.3f}")
                    fold_num += 1

                for m_name in ('balanced_accuracy', 'accuracy'):
                    collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics'][m_name] for i in range(len(predictions[cat]['folds']))])
                    if "balanced" in m_name:
                        null_value = 0.5
                    else:
                        #null_value = 1 - (1 / len(target_categories))
                        null_value = len(np.where(y[test_idx] == 1)[0]) / len(y[test_idx])
                    predictions[cat]['stats']['bayes_factors'][m_name] = tools.stats.compute_bayes(n_times, collected_scores, null_value)
                    predictions[cat]['stats']['p-values'][m_name] = tools.stats.compute_p_values(n_times, collected_scores, null_value)

                if len(target_categories) == 2:
                    m_name = "mcc"
                    collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics'][m_name] for i in range(len(predictions[cat]['folds']))])
                    null_value = 0.0
                    predictions[cat]['stats']['bayes_factors'][m_name] = tools.stats.compute_bayes(n_times, collected_scores, null_value)
                    predictions[cat]['stats']['p-values'][m_name] = tools.stats.compute_p_values(n_times, collected_scores, null_value)

            # Save
            joblib.dump(predictions, pickle_filename)
            print(f"Written '{pickle_filename}")

