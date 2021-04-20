#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil


def resampling(data, events_file, param_epoched_data, param_sfreq, param_npad, param_window,
               param_stim_picks, param_n_jobs, param_raw_pad, param_epoch_pad, 
               param_save_jointly_resampled_events):
    """Resample the signals using MNE Python and save the file once resampled.

    Parameters
    ----------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data to be resampled.
    events_file: str or None
        Path to the optional '.tsv' file containing the event matrix (2D array, shape (n_events, 3)). 
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
        if events_file is not None and param_save_jointly_resampled_events is True:

            # Convert tsv file into a numpy array of integers
            array_events = np.loadtxt(fname=events_file, delimiter="\t")
            events = array_events.astype(int)

            # Resample data
            data_resampled, events = data.resample(sfreq=param_sfreq, npad=param_npad, window=param_window,
                                                   stim_picks=param_stim_picks, n_jobs=param_n_jobs,
                                                   events=events, pad=param_raw_pad)

            # Save the events whose onsets were jointly resampled with the data
            np.savetxt("out_dir_resampling/events.tsv", array_events, delimiter="\t")

        else:
            # Resample data
            data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, window=param_window,
                                                   stim_picks=param_stim_picks, n_jobs=param_n_jobs,
                                                   events=None, pad=param_raw_pad)

    # For epoched data 
    else:

        # Resample data
        data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, 
                                       window=param_window, n_jobs=param_n_jobs, 
                                       pad=param_epoch_pad)

    # Save file
    data_resampled.save("out_dir_resampling/meg.fif", overwrite=True)

    return data_resampled 


def _compute_snr(meg_file):
    # Compute the SNR

    # select only MEG channels and exclude the bad channels
    meg_file = meg_file.pick_types(meg=True, exclude='bads')

    # create fixed length events
    array_events = mne.make_fixed_length_events(meg_file, duration=10)

    # create epochs
    epochs = mne.Epochs(meg_file, array_events)

    # mean signal amplitude on each epoch
    epochs_data = epochs.get_data()
    mean_signal_amplitude_per_epoch = epochs_data.mean(axis=(1, 2))  # mean on channels and times

    # mean across all epochs and its std error
    mean_final = mean_signal_amplitude_per_epoch.mean()
    std_error_final = np.std(mean_signal_amplitude_per_epoch, ddof=1) / np.sqrt(
        np.size(mean_signal_amplitude_per_epoch))

    # compute SNR
    snr = mean_final / std_error_final

    return snr


def _generate_report(data_file_before, raw_before_preprocessing, raw_after_preprocessing, bad_channels,
                     comments_about_filtering, notch_freqs_start, resample_sfreq, snr_before, snr_after):
    # Generate a report

    # Instance of mne.Report
    report = mne.Report(title='Results of filtering ', verbose=True)

    # Plot MEG signals in temporal domain
    fig_raw = raw_before_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, scalings='auto', butterfly=False,
                                                                          show_scrollbars=False, proj=False)
    fig_raw_maxfilter = raw_after_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, scalings='auto',
                                                                                   butterfly=False,
                                                                                   show_scrollbars=False, proj=False)

    # Plot power spectral density
    fig_raw_psd = raw_before_preprocessing.plot_psd()
    fig_raw_maxfilter_psd = raw_after_preprocessing.plot_psd()

    # Add figures to report
    report.add_figs_to_section(fig_raw, captions='MEG signals before filtering', section='Temporal domain')
    report.add_figs_to_section(fig_raw_maxfilter, captions='MEG signals after filtering',
                               comments=comments_about_filtering,
                               section='Temporal domain')
    report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before filtering',
                               section='Frequency domain')
    report.add_figs_to_section(fig_raw_maxfilter_psd, captions='Power spectral density after filtering',
                               comments=comments_about_filtering,
                               section='Frequency domain')

    # Check if MaxFilter was already applied on the data
    if raw_before_preprocessing.info['proc_history']:
        sss_info = raw_before_preprocessing.info['proc_history'][0]['max_info']['sss_info']
        tsss_info = raw_before_preprocessing.info['proc_history'][0]['max_info']['max_st']
        if bool(sss_info) or bool(tsss_info) is True:
            message_channels = f'Bad channels have been interpolated during MaxFilter'
        else:
            message_channels = bad_channels
    else:
        message_channels = bad_channels

    # Put this info in html format
    # Give some info about the file before preprocessing
    sampling_frequency = raw_before_preprocessing.info['sfreq']
    highpass = raw_before_preprocessing.info['highpass']
    lowpass = raw_before_preprocessing.info['lowpass']

    # Put this info in html format
    # Info on data
    html_text_info = f"""<html>

        <head>
            <style type="text/css">
                table {{ border-collapse: collapse;}}
                td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
            </style>
        </head>

        <body>
            <table width="50%" height="80%" border="2px">
                <tr>
                    <td>Input file: {data_file_before}</td>
                </tr>
                <tr>
                    <td>Bad channels: {message_channels}</td>
                </tr>
                <tr>
                    <td>Sampling frequency before preprocessing: {sampling_frequency}Hz</td>
                </tr>
                <tr>
                    <td>Highpass before preprocessing: {highpass}Hz</td>
                </tr>
                <tr>
                    <td>Lowpass before preprocessing: {lowpass}Hz</td>
                </tr>
            </table>
        </body>

        </html>"""

    # Info on SNR
    html_text_snr = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>SNR before filtering: {snr_before}</td>
            </tr>
            <tr>
                <td>SNR after filtering: {snr_after}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Info on SNR
    html_text_summary_filtering = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Temporal filtering: {comments_about_filtering}</td>
            </tr>
            <tr>
                <td>Notch: {notch_freqs_start}</td>
            </tr>
            <tr>
                <td>Resampling: {resample_sfreq}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_info, captions='MEG recording features', section='Data info', replace=False)
    report.add_htmls_to_section(html_text_summary_filtering, captions='Summary filtering applied',
                                section='Filtering info', replace=False)
    report.add_htmls_to_section(html_text_snr, captions='Signal to noise ratio', section='Signal to noise ratio',
                                replace=False)

    # Save report
    report.save('out_dir_report/report_filtering.html', overwrite=True)


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the files
    data_file = config.pop('fif')
    if config['param_epoched_data'] is False:
        data = mne.io.read_raw_fif(data_file, allow_maxshield=True)
    else:
        data = mne.read_epochs(data_file)

    # Read the crosstalk file
    cross_talk_file = config.pop('crosstalk')
    if os.path.exists(cross_talk_file) is True:
        shutil.copy2(cross_talk_file, 'out_dir_resampling/crosstalk_meg.fif')  # required to run a pipeline on BL

    # Read the calibration file
    calibration_file = config.pop('calibration')
    if os.path.exists(calibration_file) is True:
        shutil.copy2(calibration_file, 'out_dir_resampling/calibration_meg.dat')  # required to run a pipeline on BL

    # Read destination file 
    destination_file = config.pop('destination')
    if os.path.exists(destination_file) is True:
        shutil.copy2(destination_file, 'out_dir_resampling/destination.fif')  # required to run a pipeline on BL

    # Read head pos file
    head_pos = config.pop('headshape')
    if os.path.exists(head_pos) is True:
        shutil.copy2(head_pos, 'out_dir_resampling/headshape.pos')  # required to run a pipeline on BL

    # Read events file 
    events_file = config.pop('events')
    if os.path.exists(events_file) is False:
        events_file = None
    else:
        shutil.copy2(events_file, 'out_dir_resampling/events.tsv') # required to run a pipeline on BL

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

    # Check for None parameters

    # stim picks
    if config['param_stim_picks'] == "":
        config['param_stim_picks'] = None  # when App is run on Bl, no value for this parameter corresponds to ''  

    # Check if the user will save an empty events file 
    if events_file is None and config['param_save_jointly_resampled_events'] is True:
        value_error_message = f'You cannot save en empty events file. ' \
                              f"If you haven't an events file, please set " \
                              f"'param_save_jointly_resampled_event' to False."
        # Raise exception
        raise ValueError(value_error_message)

    # Deal with param_npad parameter

    # When the App is run on BL
    if config['param_npad'] != "auto":
        config['param_npad'] = int(config['param_npad'])

    # Deal with param_n_jobs parameter

    # When the App is run on BL
    if config['param_n_jobs'] != 'cuda':
        config['param_n_jobs']  = int(config['param_n_jobs'])

    # Deal with stim picks parameter

    # When the App is run on BL
    if isinstance(config['param_stim_picks'], str) and config['param_stim_picks'] is not None:
        config['param_stim_picks'] = config['param_stim_picks'].replace('[', '')
        config['param_stim_picks'] = config['param_stim_picks'].replace(']', '')
        config['param_stim_picks'] = config['param_stim_picks'].replace("'", '')
        config['param_stim_picks'] = list(map(int, config['param_stim_picks'].split(', ')))

    # Keep bad channels in memory
    bad_channels = data.info['bads']

    # Define kwargs
    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  

    # Apply resampling
    data_copy = data.copy()
    data_filtered = resampling(data_copy, events_file, **kwargs)
    del data_copy

    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Data was successfully resampled.'})

    # Compute SNR
    # snr_before = _compute_snr(raw)
    # snr_after = _compute_snr(raw_filtered)

    # Generate a report
    # _generate_report(data_file, raw, raw_filtered, bad_channels, comments_about_filtering,
    #                  comments_notch, comments_resample_freq, snr_before, snr_after)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
