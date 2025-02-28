#!/usr/bin/env python3

import os
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tools

def generate_plots(pickle_filename, cat_plot_filename, p_values_plot_filename, bayes_plot_filename):
    os.makedirs(os.path.dirname(cat_plot_filename), exist_ok=True)
    os.makedirs(os.path.dirname(p_values_plot_filename), exist_ok=True)
    os.makedirs(os.path.dirname(bayes_plot_filename), exist_ok=True)

    predictions = joblib.load(pickle_filename)
    print(f"Read '{pickle_filename}")

    target_categories = ['natural', 'human_made']
    if len(target_categories) == 2:
        # If it is a binary classification, only one class needs to be model
        model_categories = [target_categories[0]]
    else:
        model_categories = target_categories

    plt.figure(figsize=(10, 6))
    for cat in model_categories:
        times = predictions[cat]['folds'][0]['times']
        collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics']['balanced_accuracy'] for i in range(len(predictions[cat]['folds']))])
        ylabel = "Decoding Accuracy"
        chance_level = 1. / len(target_categories)

        mean_scores = np.mean(collected_scores, axis=0)
        sem_scores = np.std(collected_scores, axis=0) / np.sqrt(len(collected_scores)) # Standard Error of the Mean (SEM)
        plt.plot(times, mean_scores, label=cat.capitalize())
        plt.fill_between(times, mean_scores - sem_scores,
                 mean_scores + sem_scores,
                 alpha=0.3, label='Standard Error of the Mean (SEM)')
    plt.axhline(chance_level, color='k', linestyle='--', label='Chance Level')
    plt.legend()
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    if np.any(mean_scores > 0.515):
        if np.any(mean_scores > 0.58):
            plt.ylim(0.48, 0.64)
        else:
            plt.ylim(0.48, 0.58)
    else:
        plt.ylim(0.48, 0.52)
    plt.title(f'Time-resolved Decoding ({model}) - Participant {participant}')
    plt.savefig(cat_plot_filename)
    print(f"Written '{cat_plot_filename}")
    plt.clf()
    plt.close()

    # Plot the decoding accuracy and Bayes factors for a selected category.
    plt.figure(figsize=(10, 6))
    m_name = "balanced_accuracy"
    for cat in model_categories:
        bayes_factors = predictions[cat]['stats']['bayes_factors'][m_name]
        plt.plot(times, bayes_factors, label=cat.capitalize(), lw=2)
    plt.axhline(3, color='k', linestyle=':', label='Moderate Evidence Threshold at 3')
    #plt.axhline(0, color='k', linestyle='--', label='Chance Level (0)')
    plt.xlabel('Time (s)')
    plt.yscale('log')
    #plt.yscale('symlog', linthresh=1e-2) # use a linear scale for small values (below linthresh=1e-2) and a log scale for large values (so that zero is represented)
    plt.ylabel('Bayes Factor (log scale)')
    plt.ylim(0.1, 12000)
    plt.title(f'Time-Resolved Bayes Factors ({model}) - Participant {participant}')
    plt.legend()
    plt.grid()
    plt.savefig(bayes_plot_filename)
    print(f"Written '{bayes_plot_filename}")
    plt.clf()
    plt.close()

    # Plotting the frequentist p-values.
    plt.figure(figsize=(10, 6))
    m_name = "balanced_accuracy"
    for cat in model_categories:
        p_values = predictions[cat]['stats']['p-values'][m_name]
        plt.plot(times, p_values, label=cat.capitalize(), lw=2)
    plt.axhline(0.05, color='k', linestyle='--', label='p = 0.05 threshold')
    plt.xlabel('Time (s)')
    plt.ylim(-0.05, 1.05)
    plt.ylabel('p-value')
    plt.title(f'Time-Resolved Frequentist p-values {model} - Participant {participant}')
    plt.legend()
    plt.grid()
    plt.savefig(p_values_plot_filename)
    print(f"Written '{p_values_plot_filename}")
    plt.clf()
    plt.close()

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
    # Filter sessions used for training
    selected_sessions = sessions_metadata[sessions_metadata["days_since_first_recording"] < 10].groupby("participant").head(3)
    remaining_sessions_metadata = sessions_metadata.drop(selected_sessions.index)

    for participant in range(1, 4+1):
        for session in range(1, 12+1):
            for block in ('BASE', 'EARLY', 'LATE'):
                pickle_filename = f'output/models/fitted_{model}_{block}_P{participant}_S{session:02}.pkl'
                cat_plot_filename = f"output/plots/cat_{model}_{block}_P{participant}_S{session:02}.png"
                p_values_plot_filename = f"output/plots/cat_{model}_{block}_P{participant}_S{session:02}_pvalues.png"
                bayes_plot_filename = f"output/plots/cat_{model}_{block}_P{participant}_S{session:02}_bayes.png"
                if os.path.exists(pickle_filename) and not(os.path.exists(cat_plot_filename) and os.path.exists(p_values_plot_filename) and os.path.exists(bayes_plot_filename)):
                    generate_plots(pickle_filename, cat_plot_filename, p_values_plot_filename, bayes_plot_filename)

    for participant in range(1, 4+1):
        for session in range(1, 12+1):
            for block in ('EARLY', 'LATE'):
                pickle_filename = f'output/models/predictions/predictions_{model}_{block}_P{participant}_S{session:02}.pkl'
                cat_plot_filename = f"output/plots/predictions/predictions_{model}_{block}_P{participant}_S{session:02}.png"
                p_values_plot_filename = f"output/plots/predictions/predictions_{model}_{block}_P{participant}_S{session:02}_pvalues.png"
                bayes_plot_filename = f"output/plots/predictions/predictions_{model}_{block}_P{participant}_S{session:02}_bayes.png"

                if os.path.exists(cat_plot_filename) and os.path.exists(p_values_plot_filename) and os.path.exists(bayes_plot_filename):
                    continue

                if not os.path.exists(pickle_filename):
                    continue

                generate_plots(pickle_filename, cat_plot_filename, p_values_plot_filename, bayes_plot_filename)

