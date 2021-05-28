#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids
from collections import Counter


def resampling(data, events_matrix, param_epoched_data, param_sfreq, param_npad, param_window,
               param_stim_picks, param_n_jobs, param_raw_pad, param_epoch_pad, 
               param_save_jointly_resampled_events):
    """Resample the signals using MNE Python and save the file once resampled.

    Parameters
    ----------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data to be resampled.
    events_matrix: np.array or None
        The event matrix (2D array, shape (n_events, 3)). 
        When specified, the onsets of the events are resampled jointly with the data
    param_epoched_data: bool
        If True, the data to be resampled is epoched, else it is continuous.
    param_sfreq: float
        New sample rate to use in Hz.
    param_npad: int or str
        Amount to pad the start and end of the data. Can be “auto” (default).
    param_window: str
        Frequency-domain window to use in resampling. Default is "boxcar". 
    param_stim_picks: list of int or None
        Stim channels.
    param_n_jobs: int or str
        Number of jobs to run in parallel. Can be ‘cuda’ if cupy is installed properly. Default is 1.
    param_raw_pad: str
        The type of padding to use for raw data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” (default) and "edge".
    param_epoch_pad: str
        The type of padding to use for epoched data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” and "edge" (default).
    param_save_jointly_resampled_events: bool
        If True, save the events file resampled jointly with the data.

    Returns
    -------
    data_resampled: instance of mne.io.Raw or instance of mne.Epochs
        The data after resampling.
    events: array, shape (n_events, 3) or None
        If events are jointly resampled, these are returned with the raw.
        The input events are not modified.
    """

    # For continuous data 
    if param_epoched_data is False:

        # Load data
        data.load_data()

        # Test if events file exist
        if events_matrix is not None and param_save_jointly_resampled_events is True:

            # Resample data
            data_resampled, events_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, window=param_window,
                                                             stim_picks=param_stim_picks, n_jobs=param_n_jobs,
                                                             events=events_matrix, pad=param_raw_pad)

            # Save the events whose onsets were jointly resampled with the data
            # np.savetxt("out_dir_resampling/events.tsv", array_events, delimiter="\t")

        else:
            # Resample data
            data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, window=param_window,
                                           stim_picks=param_stim_picks, n_jobs=param_n_jobs,
                                           events=None, pad=param_raw_pad)
            events_resampled = None

    # For epoched data 
    else:

        # Resample data
        data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, 
                                       window=param_window, n_jobs=param_n_jobs, 
                                       pad=param_epoch_pad)
        events_resampled = None

    # Save file
    data_resampled.save("out_dir_resampling/meg.fif", overwrite=True)

    return data_resampled, events_resampled


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)


    ## Read the optional files ##

    # From meg/fif datatype #

    # Read the files
    data_file = config.pop('fif')
    if config['param_epoched_data'] is False:
        data = mne.io.read_raw_fif(data_file, allow_maxshield=True)
    else:
        data = mne.read_epochs(data_file)

    # Read the crosstalk file
    cross_talk_file = config.pop('crosstalk')

    # Read the calibration file
    calibration_file = config.pop('calibration')

    # Read destination file 
    destination_file = config.pop('destination')

    # Read head pos file
    head_pos = config.pop('headshape')
    if head_pos is not None:
        if os.path.exists(head_pos) is True:
            shutil.copy2(head_pos, 'out_dir_resampling/headshape.pos')  # required to run a pipeline on BL

    # Read the events file
    events_file = config.pop('events')
    events_file_exists = False

    # Test if events file exists
    if events_file is not None:
        if os.path.exists(events_file) is False:
            events_file = None
        else:
            events_file_exists = True
            # Warning: events file must be BIDS compliant  
            user_warning_message_events = f'The events file provided must be ' \
                                          f'BIDS compliant.'        
            warnings.warn(user_warning_message_events)
            dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_events})
            # Save events file
            shutil.copy2(events_file, 'out_dir_resampling/events.tsv')  # required to run a pipeline on BL
                                                                        # if param_save_jointly_resampled_events is True
                                                                        # it will be overwritten 


    # Read channels file 
    channels_file = config.pop('channels')
    channels_file_exists = False
    if channels_file is not None:
        if os.path.exists(channels_file):
            channels_file_exists = True
            shutil.copy2(channels_file, 'out_dir_resampling/channels.tsv')  # required to run a pipeline on BL
            df_channels = pd.read_csv(channels_file, sep='\t')
            # Select bad channels' name
            bad_channels = df_channels[df_channels["status"] == "bad"]['name']
            bad_channels = list(bad_channels.values)
            # Put channels.tsv bad channels in data.info['bads']
            data.info['bads'].sort() 
            bad_channels.sort()
            # Warning message
            if data.info['bads'] != bad_channels:
                user_warning_message_channels = f'Bad channels from the info of your data file are different from ' \
                                                f'those in the channels.tsv file. By default, only bad channels from channels.tsv ' \
                                                f'are considered as bad: the info of your data file is updated with those channels.'
                warnings.warn(user_warning_message_channels)
                dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels})
                data.info['bads'] = bad_channels


    # From meg/fif-override datatype #  

    # Read channels file
    if 'channels_override' in config.keys():
        channels_file_override = config.pop('channels_override')
        # No need to test if channels_override is None, this key is only present when the app runs on BL    
        if os.path.exists(channels_file_override) is False:
            channels_file_override = None
        else:
            if channels_file_exists:
                user_warning_message_channels_file = f"You provided two channels files: by default, the file written by " \
                                                     f"the App detecting bad channels will be used."
                warnings.warn(user_warning_message_channels_file)
                dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels_file}) 
            shutil.copy2(channels_file_override, 'out_dir_resampling/channels.tsv')  # required to run a pipeline on BL        
            df_channels = pd.read_csv(channels_file_override, sep='\t')
            # Select bad channels' name
            bad_channels_override = df_channels[df_channels["status"] == "bad"]['name']
            bad_channels_override = list(bad_channels_override.values)
            # Put channels.tsv bad channels in data.info['bads']
            data.info['bads'].sort() 
            bad_channels_override.sort()
            # Warning message
            if data.info['bads'] != bad_channels_override:
                user_warning_message_channels_override = f'Bad channels from the info of your MEG file are different from ' \
                                                         f'those in the channels.tsv file. By default, only bad channels from channels.tsv ' \
                                                         f'are considered as bad: the info of your MEG file is updated with those channels.'
                warnings.warn(user_warning_message_channels_override)
                dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_channels_override})
                data.info['bads'] = bad_channels_override      

    # Read the events file
    events_file_override_exists = False
    if "events_override" in config.keys():
        events_file = config.pop('events_override')
        # Test if events file exists
        if os.path.exists(events_file) is False:
            events_file = None
        else:
            if events_file_exists:
                user_warning_message_events_file = f"You provided two events files: by default, the file written by " \
                                                   f"app-get-events will be used."
                warnings.warn(user_warning_message_events_file)
                dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message_events_file}) 

            events_file_override_exists = True
            shutil.copy2(events_file, 'out_dir_resampling/events.tsv')  # required to run a pipeline on BL
                                                                        # this events file is not resampled  
                                                                        # if param_save_jointly_resampled_events is True
                                                                        # it will be overwritten  

    # Extract the matrix of events # 
    if config['param_epoched_data'] is False:
        if events_file_override_exists or events_file_exists:
            ############### TO BE TESTED ON NO RESTING STATE DATA
            # Compute the events matrix #
            df_events = pd.read_csv(events_file, sep='\t')
            
            # Extract relevant info from df_events
            samples = df_events['sample'].values
            event_id = df_events['value'].values

            # Compute the values for events matrix 
            events_time_in_sample = [data.first_samp + sample for sample in samples]
            values_of_trigger_channels = [0]*len(events_time_in_sample)

            # Create a dataframe
            df_events_matrix = pd.DataFrame([events_time_in_sample, values_of_trigger_channels, event_id])
            df_events_matrix = df_events_matrix.transpose()

            # Convert dataframe to numpy array
            events_matrix = df_events_matrix.to_numpy()
        if events_file_override_exists is False and events_file_exists is False:
            events_matrix = None  
    else:
        events_matrix = None               
        
    
    # Info message about resampling if applied
    if config['param_epoched_data'] is False:
        dict_json_product['brainlife'].append({'type': 'info', 'msg': f'Data was resampled at '
                                                                      f'{config["param_sfreq"]}. '
                                                                      f'Please bear in mind that it is generally '
                                                                      f'recommended not to epoch '
                                                                      f'downsampled data, but instead epoch '
                                                                      f'and then downsample.'})
    
    # Comment about resampling
    comments_resample_freq = f'{config["param_sfreq"]}Hz'

    # Convert all "" into None when the App runs on BL
    tmp = dict((k, None) for k, v in config.items() if v == "")
    config.update(tmp)

    # Check if the user will save an empty events file 
    if events_file is None and config['param_save_jointly_resampled_events'] is True:
        value_error_message = f'You cannot save en empty events file. ' \
                              f"If you haven't an events file, please set " \
                              f"'param_save_jointly_resampled_event' to False."
        # Raise exception
        raise ValueError(value_error_message)

    
    ## Convert parameters ##

    # Deal with param_npad parameter #
    # Convert param_npad into int if not "auto" when the App is run on BL
    if config['param_npad'] != "auto":
        config['param_npad'] = int(config['param_npad'])

    # Deal with param_n_jobs parameter #
    # Convert n jobs into int when the App is run on BL
    if config['param_n_jobs'] != 'cuda':
        config['param_n_jobs']  = int(config['param_n_jobs'])

    # Deal with stim picks parameter #
    # Convert stim picks into a list of int when the App is run on BL
    if isinstance(config['param_stim_picks'], str) and config['param_stim_picks'] is not None:
        config['param_stim_picks'] = config['param_stim_picks'].replace('[', '')
        config['param_stim_picks'] = config['param_stim_picks'].replace(']', '')
        config['param_stim_picks'] = list(map(int, config['param_stim_picks'].split(', ')))

    # Keep bad channels in memory
    bad_channels = data.info['bads']

    
    ## Define kwargs ##

    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  

    # Apply resampling
    data_copy = data.copy()
    data_resampled, events_resampled = resampling(data_copy, events_matrix, **kwargs)
    del data_copy

    ## Create BIDS compliant events file if existed ## 
    if events_resampled is not None and config['param_epoched_data'] is False:
        # Create a BIDSPath
        bids_path = BIDSPath(subject='subject',
                             session=None,
                             task='task',
                             run='01',
                             acquisition=None,
                             processing=None,
                             recording=None,
                             space=None,
                             suffix=None,
                             datatype='meg',
                             root='bids')

        # Extract event_id value #
        # to be tested when events are extracted from data
        event_id_value = list(events_resampled[:, 2])  # the third column of events corresponds to the value column of BIDS events.tsv
        id_values_occurrences = Counter(event_id_value)  # number of different events
        id_values_occurrences = list(id_values_occurrences.keys())
        trials_type = [f"events_{i}" for i in range(1, len(id_values_occurrences) + 1)]  # for trial type column of BIDS events.tsv 
        dict_event_id = dict((k, v) for k, v  in zip(trials_type, id_values_occurrences))


        # Write BIDS to create events.tsv BIDS compliant
        write_raw_bids(data, bids_path, events_data=events_resampled, event_id=dict_event_id, overwrite=True)

        # Extract events.tsv from bids path
        events_file = 'bids/sub-subject/meg/sub-subject_task-task_run-01_events.tsv'

        # Copy events.tsv in outdir
        shutil.copy2(events_file, 'out_dir_resampling/events.tsv') 

        # Info message in product.json
        dict_json_product['brainlife'].append({'type': 'info', 'msg': 'Jointly resampled events are saved in events.tsv.'})


    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Data was successfully resampled.'})


    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
