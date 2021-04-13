#!/usr/local/bin/python3

import json
import mne
import numpy as np
import os
import shutil


def resampling(data, param_epoched_data, param_sfreq, param_npad, param_window,
               param_stim_picks, param_n_jobs, param_events, param_raw_pad, param_epoch_pad):
    """Resample the signals using MNE Python and save the file once resampled.

    Parameters
    ----------
    data: instance of mne.io.Raw or instance of mne.Epochs
        Data to be resampled.
    param_epoched_data: bool
        If True, the data to be resampled is epoched, else it is continuous.
    param_sfreq: float
        New sample rate to use.
    param_npad: int or str
        Amount to pad the start and end of the data. Can be “auto” (default).
    param_window: str
        Frequency-domain window to use in resampling. 
    param_stim_picks: list of int or None
        Stim channels.
    param_n_jobs: int
        Number of jobs to run in parallel.
    param_events: 2D array, shape (n_events, 3) or None
        An optional event matrix. 
    param_raw_pad: str
        The type of padding to use for raw data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” (default).
    param_epoch_pad: str
        The type of padding to use for epoched data. Supports all numpy.pad() mode options. Can also be 
        “reflect_limited” or "edge" (default).

    Returns
    -------
    raw_filtered: instance of mne.io.Raw or instance of mne.Epochs
        The raw data after resampling.
    """

    # For continuous data 
    if param_epoched_data is False:

        # Load data
        data.load_data()

        # Resample data
        data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, window=param_window,
                                       stim_picks=param_stim_picks, n_jobs=param_n_jobs,
                                       events=param_events, pad=param_raw_pad)

    # For epoched data 
    else:

        # Resample data
        data_resampled = data.resample(sfreq=param_sfreq, npad=param_npad, 
                                       window=param_window, n_jobs=param_n_jobs, 
                                       pad=param_epoch_pad)

    # Save file
    data_resampled .save("out_dir_resampling/meg.fif", overwrite=True)

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

    # Read events file 
    events_file = config.pop('events')
    if os.path.exists(events_file) is True:
        shutil.copy2(events_file, 'out_dir_resampling/events.tsv')  # required to run a pipeline on BL

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

    if config['param_events'] == "":
        config['param_events'] = None  # when App is run on Bl, no value for this parameter corresponds to '' 
            
    # Keep bad channels in memory
    bad_channels = data.info['bads']

    # Define kwargs
    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  

    # Apply temporal filtering
    data_copy = data.copy()
    data_filtered = resampling(data_copy, **kwargs)
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
