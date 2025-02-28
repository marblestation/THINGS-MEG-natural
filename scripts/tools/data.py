import numpy as np
import pandas as pd
from . import io

n_sessions                  = 12
n_participants              = 4

def read_embeddings():
    embeddings_values = np.loadtxt("embeddings/spose_embedding_66d_sorted.tsv", delimiter="\t") # 1854 categories x 66 dimensions
    embeddings_dimensions = pd.read_csv("embeddings/labels_short.txt", names=["label"]) # 66 labels, one for each embedding dimension
    embeddings = pd.DataFrame(embeddings_values, columns = embeddings_dimensions['label'])

    ##colors = ('white', 'red', 'black', 'sand-colored', 'green', 'yellow', 'orange')
    #colors = ('white', 'red', 'black', 'green', 'yellow', 'orange') # do not consider 'sand-colored' given its ambiguity (it could be texture based?), which leads to an increase in other categories, especially white, yellow and orange
    #thresholds = {color: float(np.percentile(df[color], 75)) for color in colors}
    #embeddings['candidate'] = 0
    #for c in colors:
    #    embeddings.loc[embeddings[c] >= thresholds[c], "candidate"] += 1
    #
    #embeddings['selected'] = False
    #for c in colors:
    #    embeddings.loc[(embeddings['candidate'] == 1) & (embeddings[c] >= thresholds[c]), "selected"] = True
    #
    #counts = {c: len(embeddings[(embeddings['selected']) & (embeddings[c] >= thresholds[c])]) for c in colors}
    ## {'white': 154, 'red': 106, 'black': 171, 'green': 129, 'yellow': 89, 'orange': 73}

    return embeddings


def read_categories():
    categories = pd.read_excel("images/category_list.xlsx", engine="openpyxl") # 1854 rows with ['Concept number', 'Word', 'uniqueID'] columns
    high_level_categories = pd.read_csv("images/high_level_categories_wide.tsv", sep="\t") # 1855 categories x 55 columns ('uniqueID', 'Word', + 53 high-level categories with a 0 or 1 to signal belonging, one category can belong to more than one high-level category)
    categories = categories.merge(high_level_categories, on="uniqueID", how="left")

    #--------------------------------------------------------------------------------
    # -- Supra-categories
    # Human-made
    human_made = [
        'arts and crafts supply', 'clothing', 'clothing accessory',
        'construction equipment', 'container',
        'electronic device', 'fastener', 'footwear', 'furniture', 'game',
        'garden tool', 'hardware', 'headwear', 'home appliance', 'home decor',
        'jewelry', 'kitchen appliance', 'kitchen tool', 'lighting',
        'medical equipment', 'musical instrument', 'office supply', 'outerwear',
        'part of car', 'personal hygiene item', 'protective clothing',
        'safety equipment', 'school supply', 'scientific equipment',
        'sports equipment', 'tool', 'toy', 'vehicle', 'watercraft', 'weapon',
        "women's clothing",
    ]

    # Natural categories
    natural = [
        'animal', 'bird', 'farm animal', 'fruit', 'insect', 'mammal', 'plant',
        'sea animal', 'seafood', 'vegetable',
    ]

    # Unclassified (not fitting neatly into human-made or natural)
    uncategorized = ['body part', 'breakfast food', 'candy', 'food', 'dessert', 'drink', 'condiment']
    #--------------------------------------------------------------------------------

    categories['human_made'] = 0
    rows_with_human_made_only = categories.index[
        categories[human_made].eq(1).any(axis=1) & ~categories[natural].eq(1).any(axis=1)
    ].tolist()
    categories.loc[rows_with_human_made_only, 'human_made'] = 1
    #
    categories['natural'] = 0
    rows_with_natural_only = categories.index[
        categories[natural].eq(1).any(axis=1) & ~categories[human_made].eq(1).any(axis=1)
    ].tolist()
    categories.loc[rows_with_natural_only, 'natural'] = 1
    return categories

def enrich(events, categories):
    # Merge events with high-level categories
    events_df = pd.DataFrame(events[:, 2], columns=["Concept number"])
    events_df = events_df.merge(categories, on="Concept number", how="left")
    return events_df

def filter(data, events, events_df):
    keep = events_df.index[
        ((events_df['human_made'] == 1) & (events_df['natural'] == 0))
        | ((events_df['human_made'] == 0) & (events_df['natural'] == 1))
    ].tolist()
    data = data[keep]
    events = events[keep]
    events_df = events_df.iloc[keep]
    events_df = events_df.reset_index(drop=True)
    return data, events, events_df

def homogenize(data, events, events_df, categories):
    """
    Trial homogenization across multiple categories.

    For each category in `categories`, this function:
      - Selects rows where the category column equals 1 and all other category columns equal 0.
      - Determines the smallest number of trials across the categories.
      - Randomly samples each category’s rows down to that minimum number.
      - Subsets data, events, and events_df based on the selected rows.

    Parameters:
        data: An array-like or similar object indexable by row indices.
        events: An array-like or similar object indexable by row indices.
        events_df: A pandas DataFrame that includes the category columns.
        categories: A list of column names (strings) corresponding to the categories.

    Returns:
        A tuple (data, events, events_df) containing the homogenized data.
    """
    # Dictionary to store indices for each category where only that category is active
    category_indices = {}
    for cat in categories:
        # Start with rows where the current category flag is 1.
        condition = (events_df[cat] == 1)
        # Enforce exclusivity: all other categories must be 0.
        for other_cat in categories:
            if other_cat != cat:
                condition &= (events_df[other_cat] == 0)
        category_indices[cat] = events_df.index[condition].tolist()

    # Determine the minimum count across all categories.
    min_trials = min(len(idxs) for idxs in category_indices.values())

    # For each category, sample down to the minimum number of trials if necessary.
    selected_indices = []
    for cat, indices in category_indices.items():
        if len(indices) > min_trials:
            np.random.seed(42)
            sampled = np.random.choice(indices, size=min_trials, replace=False)
        else:
            sampled = indices
        selected_indices.extend(sampled)

    # Combine the indices and subset the data.
    selected_indices = np.array(selected_indices)
    if len(events) != len(selected_indices):
        print(f"Homogenization reduced the # of trials from {len(events)} to {len(selected_indices)}")
    data = data[selected_indices]
    events = events[selected_indices]
    events_df = events_df.iloc[selected_indices].reset_index(drop=True)

    return data, events, events_df

def read_metadata(participant, session):
    # Sex
    meta = pd.read_csv("THINGS-MEG/participants.tsv", sep="\t")
    meta["participant_id"] = meta["participant_id"].str.extract(r'(\d+)').astype(int)
    sex = meta.loc[meta['participant_id'] == 1, 'sex'][0]
    # Measurement Date
    input_filename = f"./output/preprocessed/preprocessed_P{participant}_S{session:02}.h5"
    data, ch_names, ch_types, sampling_rate, description, events, event_id = io.read_h5(input_filename, return_data=False)
    session_epochs = io.data2mne(data, ch_names, ch_types, sampling_rate, description, events=events, event_id=event_id, verbose=False)
    recording_datetime = session_epochs.info['meas_date']
    return recording_datetime, sex

def get_sessions():
    results = []
    first_session = {}
    for participant in range(1, n_participants+1):
        for session in range(1, n_sessions+1):
            recording_datetime, sex = read_metadata(participant, session)
            if participant not in first_session:
                first_session[participant] = recording_datetime
            days_since_first_recording = (recording_datetime - first_session[participant]).days + round((recording_datetime - first_session[participant]).seconds / 86400) if (recording_datetime - first_session[participant]).seconds != 0 else 0
            results.append((participant, sex, session, recording_datetime, days_since_first_recording))
    df = pd.DataFrame(results, columns=["participant", "sex", "session", "recording_datetime", "days_since_first_recording"])
    return df

def get_block_sessions(participant, block):
    """Returns a list of session numbers selected for a given participant and block"""
    assert block in ('BASE', 'EARLY', 'LATE')
    consider_n_sessions = 3
    if block == "BASE":
        session = 1
    elif block == "EARLY":
        session = 5
    elif block == "LATE":
        if participant == 1:
            session = 10
        elif participant == 2:
            session = 9
        elif participant == 3:
            session = 10
        elif participant == 4:
            session = 10
    sessions = [session + i for i in range(consider_n_sessions)]
    return sessions


def read_meg(participants, categories, target_categories, homogenize=False):
    all_data = None
    all_events = None
    all_event_id = None
    for participant, sessions in participants.items():
        for session in sessions:
            input_filename = f"./output/preprocessed/preprocessed_P{participant}_S{session:02}.h5"
            data, ch_names, ch_types, sampling_rate, description, events, event_id = io.read_h5(input_filename, return_data=True) # data with 2254 trials, 271 channels, 281 data points
                                                                                                                                  # events (same size as #trials) with 'Concept number' in events[:, 2]
            print(f"Read '{input_filename}' ['{len(data)} records]")
            selected = events[:, 2] <= 1854 # Ignore test and catch trials
            data = data[selected]
            events = events[selected]
            events_df = enrich(events, categories)
            data, events, events_df = filter(data, events, events_df)
            if homogenize:
                data, events, events_df = homogenize(data, events, events_df, target_categories)
            if all_data is None:
                all_data = data
                all_events = events
                all_event_id = event_id
            else:
                all_data = np.vstack((all_data, data))
                all_events = np.vstack((all_events, events))
                all_event_id.update(event_id)
            del data
            del events
    events = all_events
    data = all_data
    return data, ch_names, ch_types, sampling_rate, description, events, event_id

