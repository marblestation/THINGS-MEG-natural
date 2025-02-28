import os
import argparse
import numpy as np
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    accuracy_time_series = {}
    m_name = "balanced_accuracy"
    for participant in range(1, 4+1):
        collected_data = defaultdict(list)
        for session in range(1, 12+1):
            for origin, block in (('TRAIN', 'BASE'), ('PREDICT', 'EARLY'), ('PREDICT', 'LATE')):
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
                        collected_scores = np.asarray([predictions[cat]['folds'][i]['metrics']['balanced_accuracy'] for i in range(len(predictions[cat]['folds']))])
                        mean_scores = np.mean(collected_scores, axis=0)
                        sem_scores = np.std(collected_scores, axis=0) / np.sqrt(len(collected_scores)) # Standard Error of the Mean (SEM)
                        collected_data[cat].append((block, times, mean_scores, sem_scores))

        gif_plot_filename = f"output/plots/evolution/evolution_{model}_P{participant}.gif"
        os.makedirs(os.path.dirname(gif_plot_filename), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set up the plot
        ylabel = "Decoding Accuracy"
        chance_level = 1. / len(target_categories)
        ax.axhline(chance_level, color='k', linestyle='--', label='Chance Level')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.48, 0.64)
        ax.set_title(f'Time-resolved Decoding ({model}) - Participant {participant}')
        ax.grid(True)

        # Prepare data for animation
        all_lines_data = []
        for cat, data in collected_data.items():
            for block, times, mean_scores, sem_scores in data:
                all_lines_data.append((cat, block, times, mean_scores, sem_scores))

        # Create the animation frames
        def animate(frame):
            # Clear the previous frame
            ax.clear()

            # Re-add the constant elements
            ax.axhline(chance_level, color='k', linestyle='--', label='Chance Level')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(0.48, 0.64)
            ax.set_title(f'Time-resolved Decoding ({model}) - Participant {participant}')
            ax.grid(True)

            # Only show data up to the current frame index
            for i in range(min(frame + 1, len(all_lines_data))):
                cat, block, times, mean_scores, sem_scores = all_lines_data[i]
                #ax.plot(times, mean_scores, label=f"{cat.capitalize()} ({block})")
                ax.plot(times, mean_scores, label=f"{block}")
                ax.fill_between(times,
                              mean_scores - sem_scores,
                              mean_scores + sem_scores,
                              alpha=0.3)

            # Add legend
            ax.legend()

            return ax.get_children()

        # Create animation
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(all_lines_data),
            interval=1000, # 1 second between frames
            blit=True
        )

        # Save as GIF
        ani.save(gif_plot_filename, writer='pillow', fps=1)
        print(f"Written '{gif_plot_filename}'")
        plt.close()
