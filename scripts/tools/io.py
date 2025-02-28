import os
import numpy as np
import json
import pickle
import base64
import shutil
import tempfile
import h5py
import mne
from mne import Info
from mne.io.constants import FIFF

CH_TYPE2FIFF = {
    "meg": FIFF.FIFFV_MEG_CH,
    "mag": FIFF.FIFFV_MEG_CH, # ?
    "eeg": FIFF.FIFFV_EEG_CH,
    "emg": FIFF.FIFFV_EMG_CH,
    "eog": FIFF.FIFFV_EOG_CH,
    "ecg": FIFF.FIFFV_ECG_CH,
    "ecog": FIFF.FIFFV_ECOG_CH,
    "brain_region": FIFF.FIFFV_ECOG_CH, # ECoG (ecog): Electrocorticography channels, recording electrical activity directly from the brain's surface
    "misc": FIFF.FIFFV_MISC_CH,
}

DATA_CLASSES = ('RawArray', 'EpochsArray', 'EvokedArray', 'TimeCourses', 'EpochsTimeCourses', 'AverageTimeCourses', 'SpectroTemporalConnectivity', 'SpectralConnectivity', 'EpochsSpectralConnectivity', 'EpochsTFR', 'AverageTFR', )

def validate_data(data, ch_names, ch_types, sampling_rate, description, events=None, event_id=None):
    assert "class" in description
    class_name = description['class']
    assert class_name in DATA_CLASSES

    if len(ch_names) != len(ch_types):
        raise Exception(f"Number of channels ({len(ch_names)}) and types ({len(ch_types)}) do not match!")

    epochs_required_params = ('tmin', 'tmax', 'tstart', 'filter_length', 'tolerate_overlap_fraction',)
    connectivity_required_params = ('connectivity_methods', 'frequency_bands',)
    time_frequency_required_params = ('n_cycles', 'n_frequencies', 'frequencies')
    if class_name == 'RawArray':
        # Continuous Time Data
        # https://mne.tools/stable/generated/mne.io.RawArray.html
        ## (n_channels, n_timesteps)
        if len(data.shape) != 2:
            raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
        n_channels = data.shape[0]
        if len(ch_names) != n_channels:
            raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
        if events is not None or event_id is not None:
            raise Exception(f"Event/event_id is provided but data does not have epochs")
    elif class_name in ('EpochsArray', 'EvokedArray'):
        # Epoched Data
        if class_name == 'EpochsArray':
            # https://mne.tools/stable/generated/mne.EpochsArray.html#mne.EpochsArray.save
            ## (n_epochs, n_channels, n_timesteps)
            ## First dimension: 0 = avg_power; 1 = itc
            if len(data.shape) != 3:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_epochs = data.shape[0]
            #--------------------------------------------------------------------------------
            if events is None or event_id is None:
                raise Exception(f"Data structure with epochs but events/event_id is missing")
            if events.shape[0] != n_epochs:
                raise Exception(f"Number of events ({events.shape[0]}) and number of epochs ({n_epochs}) in the data do not match")
            for required_param in epochs_required_params:
                if required_param not in description:
                    raise Exception(f"Data structure with epochs but '{required_param}' is missing from the description")
            event_ids_in_data = set(events[:, 2])
            event_ids_provided = set(event_id.values())
            if len(event_ids_in_data.symmetric_difference(event_ids_provided)) > 0:
                raise Exception(f"Event ids in data ({event_ids_in_data}) and provided event ids ({event_ids_provided}) do not match")
            #--------------------------------------------------------------------------------
        elif class_name == 'EvokedArray':
            # Averaged epochs
            # https://mne.tools/stable/generated/mne.EvokedArray.html#mne.EvokedArray.save
            ## (n_channels, n_timesteps)
            if len(data.shape) != 2:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[0]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
    elif class_name in ('TimeCourses', 'EpochsTimeCourses', 'AverageTimeCourses'):
        # Source Reconstructed
        if class_name == "EpochsTimeCourses":
            ## (n_epochs, n_labels, n_timesteps)
            if len(data.shape) != 3:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_epochs = data.shape[0]
            #--------------------------------------------------------------------------------
            if events is None or event_id is None:
                raise Exception(f"Data structure with epochs but events/event_id is missing")
            if events.shape[0] != n_epochs:
                raise Exception(f"Number of events ({events.shape[0]}) and number of epochs ({n_epochs}) in the data do not match")
            for required_param in epochs_required_params:
                if required_param not in description:
                    raise Exception(f"Data structure with epochs but '{required_param}' is missing from the description")
            event_ids_in_data = set(events[:, 2])
            event_ids_provided = set(event_id.values())
            if len(event_ids_in_data.symmetric_difference(event_ids_provided)) > 0:
                raise Exception(f"Event ids in data ({event_ids_in_data}) and provided event ids ({event_ids_provided}) do not match")
            #--------------------------------------------------------------------------------
        elif class_name == "AverageTimeCourses":
            ## (n_labels, n_timesteps)
            if len(data.shape) != 2:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[0]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
        elif class_name == "TimeCourses":
            ### Raw EEG not epoched
            ## (n_labels, n_timesteps)
            if len(data.shape) != 2:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[0]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
    elif class_name in ('SpectroTemporalConnectivity', 'SpectralConnectivity', 'EpochsSpectralConnectivity',):
        # Spectral Connectivity
        ##   connectivity_matrix_per_band_and_epoch_array            => (n_connectivity_methods, n_epochs, n_labels, n_labels,       n_freq_bands)  => EpochsSpectralConnectivity
        ##   connectivity_matrix_per_band_and_timestep_array         => (n_connectivity_methods, n_labels, n_labels, n_freq_bands,   n_times     )  => SpectroTemporalConnectivity
        ##   connectivity_matrix_per_band_array                      => (n_connectivity_methods, n_labels, n_labels, n_freq_bands                )  => SpectralConnectivity

        if class_name == "EpochsSpectralConnectivity":
            n_frequency_bands = data.shape[4]
            if len(data.shape) != 5:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[2]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_channels = data.shape[3]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_epochs = data.shape[1]
            #--------------------------------------------------------------------------------
            if events is None or event_id is None:
                raise Exception(f"Data structure with epochs but events/event_id is missing")
            if events.shape[0] != n_epochs:
                raise Exception(f"Number of events ({events.shape[0]}) and number of epochs ({n_epochs}) in the data do not match")
            for required_param in epochs_required_params:
                if required_param not in description:
                    raise Exception(f"Data structure with epochs but '{required_param}' is missing from the description")
            event_ids_in_data = set(events[:, 2])
            event_ids_provided = set(event_id.values())
            if len(event_ids_in_data.symmetric_difference(event_ids_provided)) > 0:
                raise Exception(f"Event ids in data ({event_ids_in_data}) and provided event ids ({event_ids_provided}) do not match")
            #--------------------------------------------------------------------------------
        elif class_name == "SpectroTemporalConnectivity":
            n_frequency_bands = data.shape[3]
            if len(data.shape) != 5:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_channels = data.shape[2]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
        elif class_name == "SpectralConnectivity":
            n_frequency_bands = data.shape[3]
            if len(data.shape) != 4:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_channels = data.shape[2]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
        # Common to all connectivity data structures:
        for required_param in connectivity_required_params:
            if required_param not in description:
                raise Exception(f"Data structure with connectivity but '{required_param}' is missing from the description")
        connectivity_methods = description['connectivity_methods']
        n_connectivity_methods = data.shape[0]
        if len(connectivity_methods) != n_connectivity_methods:
            raise Exception(f"Number of connectivity methods ({len(connectivity_methods)}) and number of rows ({n_connectivity_methods}) in the data do not match")
        waves_freq_ranges = json.loads(description['frequency_bands'])
        if len(waves_freq_ranges) != n_frequency_bands:
            raise Exception(f"Number of frequency bands ({len(waves_freq_ranges)}) and number of rows ({data.shape[3]}) in the data do not match")
    elif class_name in ('EpochsTFR', 'AverageTFR', ):
        # Time-frequency rerpresentation
        if class_name == 'EpochsTFR':
            ## (n_epochs, n_eeg_channels, n_freq, n_power_times)
            if len(data.shape) != 4:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            n_epochs = data.shape[0]
            #--------------------------------------------------------------------------------
            if events is None or event_id is None:
                raise Exception(f"Data structure with epochs but events/event_id is missing")
            if events.shape[0] != n_epochs:
                raise Exception(f"Number of events ({events.shape[0]}) and number of epochs ({n_epochs}) in the data do not match")
            for required_param in epochs_required_params:
                if required_param not in description:
                    raise Exception(f"Data structure with epochs but '{required_param}' is missing from the description")
            event_ids_in_data = set(events[:, 2])
            event_ids_provided = set(event_id.values())
            if len(event_ids_in_data.symmetric_difference(event_ids_provided)) > 0:
                raise Exception(f"Event ids in data ({event_ids_in_data}) and provided event ids ({event_ids_provided}) do not match")
            #--------------------------------------------------------------------------------
        elif class_name == "AverageTFR":
            #stacked_data.shape  => (2, n_eeg_channels, n_freqs, n_power_times)
            if len(data.shape) != 4:
                raise Exception(f"Data class '{class_name}' cannot have '{len(data.shape)}' dimensions")
            if data.shape[0] != 2:
                raise Exception(f"Data class '{class_name}' cannot have '{data.shape[0]}' elements in dimension zero (two are expected)")
            n_channels = data.shape[1]
            if len(ch_names) != n_channels:
                raise Exception(f"Number of channels ({len(ch_names)}) and number of rows ({n_channels}) in the data do not match")
            if events is not None or event_id is not None:
                raise Exception(f"Event/event_id is provided but data does not have epochs")
        # Common to all time-frequency data structures:
        for required_param in time_frequency_required_params:
            if required_param not in description:
                raise Exception(f"Data structure with connectivity but '{required_param}' is missing from the description")

def get_montage(device):
    montage = None
    if device in ("ANT-Neuro-64", "g.tec-64"):
        ### g.tec-64
        ### ANT eego: https://www.ant-neuro.com/sites/default/files/images/waveguard_layout_064ch.png
        ## Set 10-20 standard montage
        montage = mne.channels.make_standard_montage('standard_1020')
        #--------------------------------------------------------------------------------
        ### Validate that MNE standard 1020 matches ANT64 positions
        ##ant64 = pymatreader.read_mat("input/ant64_elec.mat")
        ##ant64_pos = ant64['elec']['elecpos']
        ##ant64_labels = ant64['elec']['label']
        ##scale_factor = 1000. # from mm to m
        ##pos = {label: [x/scale_factor, y/scale_factor, z/scale_factor] for label, (x, y, z) in zip(ant64_labels, ant64_pos)}
        ##ant64_montage = mne.channels.make_dig_montage(pos, coord_frame="head")
        ##for k, v in ant64_montage.get_positions()['ch_pos'].items():
            ##if np.any((montage.get_positions()['ch_pos'][k] - v) != 0):
                ##raise Exception(f"Different positions for '{k}'!")
        ##import matplotlib.pyplot as plt
        ##fig = ant64_montage.plot(sphere="auto")
        ### https://www.ant-neuro.com/sites/default/files/images/waveguard_layout_064ch.png
        ##fig.savefig("output/ANT_waveguard64_montage.png")
        ##fig.clf()
        ##plt.close(fig)
        #--------------------------------------------------------------------------------
    elif device == "ANT-Neuro-128-Duke":
        ### ANT eego duke: https://www.ant-neuro.com/sites/default/files/images/waveguard_duke_128ch.png
        ant_montage_filename = "input/ANT_waveguard128_duke_montage.xyz"
        montage = get_ant_montage(ant_montage_filename)
    elif device == 'I-CARE':
        ## Derive a new montage from a standard 10-20 but with only the required electrodes
        #required_ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        #base_montage = mne.channels.make_standard_montage('standard_1020')
        #details = base_montage.get_positions()
        #ch_pos = {k: p for k, p in details['ch_pos'].items() if k in required_ch_names}
        #montage = mne.channels.make_dig_montage(ch_pos, nasion=details['nasion'], lpa=details['lpa'], rpa=details['rpa'], hsp=details['hsp'], hpi=details['hpi'], coord_frame=details['coord_frame'])
        montage = mne.channels.make_standard_montage('standard_1020')
    return montage

def pickle_to_text(obj):
    """Serialize an object to a Base64-encoded string using pickle."""
    pickled_bytes = pickle.dumps(obj)  # Pickle the object (binary)
    encoded_str = base64.b64encode(pickled_bytes).decode('utf-8')  # Convert to Base64 string
    return encoded_str

def text_to_pickle(encoded_str):
    """Deserialize a Base64-encoded pickle string back into an object."""
    pickled_bytes = base64.b64decode(encoded_str)  # Decode Base64
    obj = pickle.loads(pickled_bytes)  # Unpickle back to original object
    return obj

def mne2data(mne_array):
    """                                                                                                                                                                                                         mne.io.RawArray or mne.EpochsArray to data                                                                                                                                                                  """
    if mne_array.info['description'] is None:
        description = {'class': mne_array.__class__.__name__}
    else:
        description = json.loads(mne_array.info['description'])
    mne_info = dict(mne_array.info)
    for field in ('description', 'ch_names', 'sfreq'):
        mne_info.pop(field, None)
    # Serialize MNE Info data using pickle and Base64 encoding
    description['mne'] = pickle_to_text(mne_info)
    ch_names = mne_array.info['ch_names']
    ch_types = mne_array.info.get_channel_types()
    ch_types = [t if t != "ecog" else "brain_region" for t in ch_types] # ECoG (ecog): Electrocorticography channels, recording electrical activity directly from the brain's surface
    sampling_rate = mne_array.info['sfreq']
    data = mne_array._data
    events = None
    event_id = None
    if 'events' in dir(mne_array) and 'event_id' in dir(mne_array):
        description['tmin'] = description.get('tmin', mne_array.times[0])
        description['tmax'] = description.get('tmax', mne_array.times[-1])
        description['tstart'] = description.get('tstart', 0)
        description['filter_length'] = description.get('filter_length', 0)
        description['tolerate_overlap_fraction'] = description.get('tolerate_overlap_fraction', 0)
        events = mne_array.events
        event_id = mne_array.event_id
    validate_data(data, ch_names, ch_types, sampling_rate, description, events=events, event_id=event_id)
    if events is not None:
        return data, ch_names, ch_types, sampling_rate, description, events, event_id
    else:
        return data, ch_names, ch_types, sampling_rate, description


def data2mne(data, ch_names, ch_types, sampling_rate, description, events=None, event_id=None, verbose=None):
    """
    data to mne.io.RawArray or mne.EpochsArray
    """
    assert "class" in description
    class_name = description['class']
    assert class_name in DATA_CLASSES
    if description.get('mne') is not None:
        # Deserialize stringo into MNE Info data using pickle and Base64 encoding
        mne_info = text_to_pickle(description['mne'])
        mne_info['sfreq'] = sampling_rate
        mne_info['ch_names'] = ch_names
        # Make sure serialized channels respect the current channel order/selection
        previous_chs = mne_info.pop('chs', [])
        mne_info['chs'] = []
        for ch_name, ch_type in zip(ch_names, ch_types):
            ch_type_fiff = CH_TYPE2FIFF[ch_type]
            for ch in previous_chs:
                if ch['ch_name'] == ch_name and ch['kind'] == ch_type_fiff:
                    mne_info['chs'].append(ch)
                    break
        info = Info(mne_info)
    else:
        ch_types = [t if t != "brain_region" else "ecog" for t in ch_types] # ECoG (ecog): Electrocorticography channels, recording electrical activity directly from the brain's surface
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=ch_types,
            sfreq=sampling_rate
        )
    with info._unlock():
        if 'higher_frequency' in description:
            info["lowpass"] = description['higher_frequency']
        if 'higher_frequency' in description:
            info["highpass"] = description["lower_frequency"]
        if description.get('reference_channel', description.get('online_reference_channel')) == 'average':
            info['custom_ref_applied'] = True
    info['description'] = json.dumps(description)
    tmin = description.get('tmin', 0)
    nave = description.get('n_averaged_epochs', 0)
    if class_name == 'RawArray':
        assert len(data.shape) == 2
        assert nave == 0
        mne_array = mne.io.RawArray(data, info, verbose=verbose)
    elif class_name == 'EpochsArray':
        assert len(data.shape) == 3
        assert nave == 0
        mne_array = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id, on_missing='warn', verbose=verbose)
    elif class_name == 'EvokedArray':
        assert len(data.shape) == 2
        assert nave > 0
        mne_array = mne.EvokedArray(data, info, tmin=tmin, nave=nave)
    elif class_name in ('EpochsTFR', 'AverageTFR',):
        freqs = json.loads(description['frequencies'])
        #times = np.arange(description['tmin'], description['tmax'], (description['tmax']-description['tmin'])/data.shape[-1])
        times = np.linspace(description['tmin'], description['tmax'], data.shape[-1])
        if class_name == 'EpochsTFR':
            mne_array = mne.time_frequency.EpochsTFRArray(info, data, times, freqs)
        elif class_name == 'AverageTFR':
            mne_array = mne.time_frequency.AverageTFRArray(info, data, times, freqs)
    elif class_name == 'EpochsTimeCourses':
        assert len(data.shape) == 3
        mne_array = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id, on_missing='warn')
    elif class_name == 'AverageTimeCourses':
        assert len(data.shape) == 2
        assert nave > 0
        mne_array = mne.EvokedArray(data, info, tmin=tmin, nave=nave)
    is_source_reconstruction = any(["Source reconstruction" in step for step in description.get('history', [])])
    if not is_source_reconstruction:
        device = description.get('device')
        montage = get_montage(device)
        if class_name in ('RawArray', 'EpochsArray', 'EvokedArray',):
            if montage is not None:
                mne_array.set_montage(montage)
        else:
            mne_array.info.set_montage(montage)
    return mne_array

def write_h5(output_filename, data, ch_names, ch_types, sampling_rate, description, events=None, event_id=None):
    """Write the h5 file in a temporary directory and move after it is fully written to reduce the risk of half-writing a file"""
    validate_data(data, ch_names, ch_types, sampling_rate, description, events=events, event_id=event_id)

    output_dirname = os.path.dirname(output_filename)
    with tempfile.TemporaryDirectory(dir=output_dirname) as tmp_dirname: # tmp_dirname will be removed if an exception occurs
        tmp_filename = os.path.join(tmp_dirname, os.path.basename(output_filename)) # Use same name as target otherwise mne cannot read splitted files that have been renamed

        with h5py.File(tmp_filename, 'w') as h5file:
            dt = h5py.special_dtype(vlen=str)
            h5file.create_dataset('ch_names', data=ch_names, dtype=dt)
            h5file.create_dataset('ch_types', data=ch_types, dtype=dt)
            h5file.create_dataset('data', data=data)
            if events is not None:
                h5file.create_dataset('events', data=events)
            h5file.attrs['sampling_rate'] = sampling_rate
            if event_id is not None:
                h5file.attrs['event_id'] = json.dumps(event_id)
            for key, value in description.items():
                h5file.attrs[key] = value

        shutil.move(tmp_filename, output_dirname)
    del data

def write_fif(raw_array, output_filename):
    """Write the raw array in a temporary directory and move after it is fully written to reduce the risk of half-writing a file"""
    output_dirname = os.path.dirname(output_filename)
    with tempfile.TemporaryDirectory(dir=output_dirname) as tmp_dirname: # tmp_dirname will be removed if an exception occurs
        tmp_filename = os.path.basename(output_filename) # Use same name as target otherwise mne cannot read splitted files that have been renamed
        raw_array.save(os.path.join(tmp_dirname, tmp_filename), fmt='double', overwrite=True)
        # The save method can create multiple *.fif, *-1.fif, -*-2.fif files depending on the array size
        # It is necessary to move all of them:
        tmp_files = sorted([f for f in os.listdir(tmp_dirname) if f.endswith(".fif")])
        for i, tmp_file in enumerate(tmp_files):
            if i == 0:
                final_filename = output_filename
            else:
                final_filename = output_filename.replace('.fif', f'-{i}.fif')
            shutil.move(os.path.join(tmp_dirname, tmp_file), os.path.join(output_dirname, final_filename))
        del raw_array


def read_h5(h5_filename, return_data=True):
    with h5py.File(h5_filename, 'r') as h5file:
        # Retrieve the serialized object
        ch_names = [s.decode('utf-8') for s in h5file['ch_names'][:]]
        ch_types = [s.decode('utf-8') for s in h5file['ch_types'][:]]
        description = dict(h5file.attrs)
        # Convert numpy numbers/arrays to python numbers/lists
        for key, value in description.items():
            if isinstance(value, (np.int32, np.int64)):
                description[key] = int(value)
            if isinstance(value, (np.float32, np.float64)):
                description[key] = float(value)
            elif isinstance(value, np.ndarray):
                description[key] = value.tolist()
        sampling_rate = description.pop('sampling_rate')
        if return_data:
            data = h5file['data'][()]
        else:
            if len(h5file['data'].shape) == 3:
                data = np.zeros((1, h5file['data'].shape[1], h5file['data'].shape[2]))
            elif len(h5file['data'].shape) == 3:
                data = np.zeros((h5file['data'].shape[0], 0))
            else:
                data = None
        events = None
        event_id = None
        if 'events' in h5file:
            has_events = True
            if return_data:
                events = h5file['events'][()]
                event_id = json.loads(description.pop('event_id'))
            else:
                events = np.zeros((1, 3), dtype=np.int64)
                event_id = {'0': 0}
        else:
            has_events = False
    if return_data:
        validate_data(data, ch_names, ch_types, sampling_rate, description, events=events, event_id=event_id)
    if not has_events:
        return data, ch_names, ch_types, sampling_rate, description
    else:
        return data, ch_names, ch_types, sampling_rate, description, events, event_id


