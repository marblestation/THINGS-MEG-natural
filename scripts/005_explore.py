#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import pearsonr
from dtw import dtw
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

    sessions_metadata = tools.data.get_sessions()
    collected_data = []
    accuracy_time_series = {}
    m_name = "balanced_accuracy"
    for participant in range(1, 4+1):
        for session in range(1, 12+1):
            for origin, block in (('TRAIN', 'BASE'), ('TRAIN', 'EARLY'), ('TRAIN', 'LATE'), ('PREDICT', 'EARLY'), ('PREDICT', 'LATE')):
                if origin == 'TRAIN':
                    pickle_filename = f'output/models/fitted_{model}_{block}_P{participant}_S{session:02}.pkl'
                else:
                    pickle_filename = f'output/models/predictions/predictions_{model}_{block}_P{participant}_S{session:02}.pkl'
                if os.path.exists(pickle_filename):
                    predictions = joblib.load(pickle_filename)
                    print(f"Read '{pickle_filename}")

                    target_categories = ['natural', 'human_made']
                    if len(target_categories) == 2:
                        # If it is a binary classification, only one class needs to be model
                        model_categories = [target_categories[0]]
                    else:
                        model_categories = target_categories

                    for cat in model_categories:
                        times = predictions[cat]['folds'][0]['times']
                        #if len(target_categories) == 2:
                        #    collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics']['mcc'] for i in range(len(predictions[cat]['folds']))])
                        #    ylabel = "Decoding MCC"
                        #    chance_level = 0
                        #else:
                        collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics']['balanced_accuracy'] for i in range(len(predictions[cat]['folds']))])
                        ylabel = "Decoding Accuracy"
                        chance_level = 1. / len(target_categories)

                        mean_scores = np.mean(collected_scores, axis=0)
                        sem_scores = np.std(collected_scores, axis=0) / np.sqrt(len(collected_scores)) # Standard Error of the Mean (SEM)
                        #sem_scores = np.std(collected_scores, axis=0)
                        bayes_factors = predictions[cat]['stats']['bayes_factors'][m_name]
                        p_values = predictions[cat]['stats']['p-values'][m_name]

                        accuracy_time_series[f"{participant}_{session}_{origin}_{block}"] = mean_scores

                        select = (times >= 0.075) & (times <= 0.975) # Area of interest
                        #select = (times >= 0.075) & (times <= 0.275) # Area of interest
                        #select = (times >= 0.275) & (times <= 0.975) # Area of interest
                        accuracy_weighted_mean = np.sum(mean_scores[select] * (1./sem_scores[select])) / np.sum(1./sem_scores[select])

                        threshold = 3
                        above_threshold = bayes_factors > threshold
                        below_threshold = bayes_factors <= threshold
                        bayes_factors_area_above = np.sum(bayes_factors[select & above_threshold] - threshold)
                        bayes_factors_area_below = np.sum(threshold - np.abs(bayes_factors[select & below_threshold]))
                        bayes_factors = np.log10(bayes_factors)
                        bayes_factors_area_above = np.log10(bayes_factors_area_above)
                        bayes_factors_area_below = np.log10(bayes_factors_area_below)
                        accuracy_mean = np.mean(mean_scores[select])
                        accuracy_std = np.std(mean_scores[select])
                        accuracy_sem = np.mean(sem_scores[select])
                        p_value_mean = np.mean(p_values[select])
                        p_value_std = np.std(p_values[select])
                        bayes_factor_mean = np.mean(bayes_factors[select])
                        bayes_factor_std = np.std(bayes_factors[select])
                        #print(f"{model}\t{participant}\t{session}\t{block}\t{accuracy_mean:.4f} +/- {accuracy_std:.4f} / {accuracy_sem:.4f} [{accuracy_weighted_mean:.4f}] {p_value_mean:.4f} +/- {p_value_std:.4f} | {bayes_factor_mean:.4f} +/- {bayes_factor_std:.4f} [{bayes_factors_area_above:.4f}] [{bayes_factors_area_below:.4f}]")


                        meta_participant = sessions_metadata.loc[(sessions_metadata['participant'] == participant)].reset_index()
                        sex = meta_participant.iloc[0]["sex"]
                        recording_datetime = []
                        days_since_first_recording = []
                        for s in [session + i for i in range(3)]:
                            meta_session = sessions_metadata.loc[(sessions_metadata['participant'] == participant) & (sessions_metadata['session'] == s)].reset_index()
                            recording_datetime.append(meta_session.iloc[0]["recording_datetime"])
                            days_since_first_recording.append(float(meta_session.iloc[0]["days_since_first_recording"]))
                        days_since_start = np.mean(days_since_first_recording)
                        block_timespan_in_days = days_since_first_recording[-1] - days_since_first_recording[0]
                        collected_data.append((model, participant, sex, session, origin, block, days_since_start, block_timespan_in_days, accuracy_mean, accuracy_std, accuracy_sem, accuracy_weighted_mean, p_value_mean, p_value_std, bayes_factor_mean, bayes_factor_std, bayes_factors_area_above, bayes_factors_area_below))

    enriched_collected_data = []
    for (model, participant, sex, session, origin, block, days_since_start, block_timespan_in_days, accuracy_mean, accuracy_std, accuracy_sem, accuracy_weighted_mean, p_value_mean, p_value_std, bayes_factor_mean, bayes_factor_std, bayes_factors_area_above, bayes_factors_area_below) in collected_data:
        if block != "BASE":
            predict_accuracy_time_serie = accuracy_time_series[f"{participant}_{session}_PREDICT_{block}"]
            train_accuracy_time_serie = accuracy_time_series[f"{participant}_{session}_TRAIN_{block}"]
            # Distances
            accuracy_euclidean_dist = euclidean(predict_accuracy_time_serie, train_accuracy_time_serie)
            accuracy_manhattan_dist = cityblock(predict_accuracy_time_serie, train_accuracy_time_serie)
            # Compute Dynamic Time Warping Distance
            accuracy_dtw_dist = dtw(predict_accuracy_time_serie, train_accuracy_time_serie).distance
            # Pearson Correlation Coefficient
            accuracy_pearson_corr, _ = pearsonr(predict_accuracy_time_serie, train_accuracy_time_serie)
        else:
            accuracy_euclidean_dist = 0
            accuracy_manhattan_dist = 0
            accuracy_dtw_dist = 0
            accuracy_pearson_corr = 0
        enriched_collected_data.append((model, participant, sex, session, origin, block, days_since_start, block_timespan_in_days, accuracy_mean, accuracy_std, accuracy_sem, accuracy_weighted_mean, p_value_mean, p_value_std, bayes_factor_mean, bayes_factor_std, bayes_factors_area_above, bayes_factors_area_below, accuracy_euclidean_dist, accuracy_manhattan_dist, accuracy_dtw_dist, accuracy_pearson_corr))

    df = pd.DataFrame(enriched_collected_data, columns=['model', 'participant', 'sex', 'session', 'origin', 'block', 'days_since_start', 'block_timespan_in_days', 'accuracy_mean', 'accuracy_std', 'accuracy_sem', 'accuracy_weighted_mean', 'p_value_mean', 'p_value_std', 'bayes_factor_mean', 'bayes_factor_std', 'bayes_factors_area_above', 'bayes_factors_area_below', 'accuracy_euclidean_dist', 'accuracy_manhattan_dist', 'accuracy_dtw_dist', 'accuracy_pearson_corr'])

    summary_filename = f"output/summary_{model}.tsv"
    os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
    df.to_csv(summary_filename, index=False, sep="\t")
    print(f"Written '{summary_filename}")





