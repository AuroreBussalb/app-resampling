#!/usr/local/bin/python3

import json
import mne
import numpy as np


def temporal_filtering(raw, param_filter_l_freq, param_filter_h_freq, param_filter_picks, param_filter_length,
                       param_filter_l_trans_bandwidth, param_filter_h_trans_bandwidth, param_filter_n_jobs,
                       param_filter_method, param_filter_iir_params, param_filter_phase, param_filter_fir_window,
                       param_filter_fir_design, param_filter_skip_by_annotation, param_filter_pad, param_apply_notch,
                       param_notch_freqs_start, param_notch_freqs_end, param_notch_freqs_step, param_notch_picks,
                       param_notch_filter_length, param_notch_widths, param_notch_trans_bandwidth, param_notch_n_jobs,
                       param_notch_method, param_notch_iir_parameters, param_notch_mt_bandwidth, param_notch_p_value,
                       param_notch_phase, param_notch_fir_window, param_notch_fir_design, param_notch_pad,
                       param_apply_resample, param_resample_sfreq, param_resample_npad, param_resample_window,
                       param_resample_stim_picks, param_resample_n_jobs, param_resample_events, param_resample_pad):
    """Perform filtering using MNE Python and save the file once filtered.

    Parameters
    ----------
    raw: instance of mne.io.Raw
        Data to be filtered.
    param_filter_l_freq: float or None
        For FIR filters, the lower pass-band edge; for IIR filters, the lower cutoff frequency. If None the 
        data are only low-passed.
    param_filter_h_freq: float or None
        For FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency. If None the 
        data are only high-passed.
    param_filter_picks: str, list, or None
        Channels to include.
    param_filter_length: str
        Length of the FIR filter to use (if applicable). Can be ‘auto’ (default) : the filter length is chosen based 
        on the size of the transition regions, or an other str (human-readable time in units of “s” or “ms”: 
        e.g., “10s” or “5500ms”). 
    param_filter_l_trans_bandwidth: float or str
        Width of the transition band at the low cut-off frequency in Hz (high pass or cutoff 1 in bandpass). 
        Can be “auto” (default) to use a multiple of l_freq.     
    param_filter_h_trans_bandwidth: float or str   
        Width of the transition band at the high cut-off frequency in Hz (low pass or cutoff 2 in bandpass). 
        Can be “auto” (default) to use a multiple of h_freq.
    param_filter_n_jobs: int
        Number of jobs to run in parallel.
    param_filter_method: str
        ‘fir’ will use overlap-add FIR filtering, ‘iir’ will use IIR forward-backward filtering (via filtfilt).
    param_filter_iir_params: dict or None
        Dictionary of parameters to use for IIR filtering. If iir_params is None and method=”iir”, 
        4th order Butterworth will be used. 
    param_filter_phase: str
        Phase of the filter, only used if method='fir'. Either 'zero' or 'zero-double'.
    param_filter_fir_window: str
        The window to use in FIR design, can be “hamming” (default), “hann” (default in 0.13), or “blackman”.
    param_filter_fir_design: str
        Can be “firwin” (default) or “firwin2”.
    param_filter_skip_by_annotation: str or list of str
        If a string (or list of str), any annotation segment that begins with the given string will not be included in
        filtering, and segments on either side of the given excluded annotated segment will be filtered separately.
    param_filter_pad: str
        The type of padding to use. Supports all numpy.pad() mode options. Can also be “reflect_limited” (default).
    param_apply_notch: bool
        If True apply a notch filter.
    param_notch_freqs_start: int
        Frequency to notch filter in Hz.
    param_notch_freqs_end: int
        End of the interval (in Hz) of the power lines harmonics to notch filter. This value is excluded.
    param_notch_freqs_step: int
        The step in Hz to filter power lines harmonics between param_notch_freqs_start and param_notch_freqs_end.
    param_notch_picks: list, or None
        Channels to include.
    param_notch_filter_length: str
        Length of the FIR filter to use (if applicable). Can be ‘auto’ (default) : the filter length is chosen based 
        on the size of the transition regions, or an other str (human-readable time in units of “s” or “ms”: 
        e.g., “10s” or “5500ms”. 
    param_notch_widths: float or None
        Width of the stop band in Hz. If None, freqs / 200 is used.
    param_notch_trans_bandwidth: float
        Width of the transition band in Hz. 
    param_notch_n_jobs: int
        Number of jobs to run in parallel.
    param_notch_method: str
        ‘fir’ will use overlap-add FIR filtering, ‘iir’ will use IIR forward-backward filtering (via filtfilt). 
    param_notch_iir_parameters: dict or None
        Dictionary of parameters to use for IIR filtering. If iir_params is None and method=”iir”, 
        4th order Butterworth will be used.
    param_notch_mt_bandwidth: float or None
        The bandwidth of the multitaper windowing function in Hz.
    param_notch_p_value: float
        P-value to use in F-test thresholding to determine significant sinusoidal components 
        to remove when method=’spectrum_fit’ and freqs=None.
    param_notch_phase: str
        Phase of the filter, only used if method='fir'. Either 'zero' or 'zero-double'.
    param_notch_fir_window: str
        The window to use in FIR design, can be “hamming” (default), “hann”, or “blackman”.
    param_notch_fir_design: str
        Can be “firwin” (default) or “firwin2”.
    param_notch_pad: str
        The type of padding to use. Supports all numpy.pad() mode options. Can also be “reflect_limited” (default).
    param_apply_resample: bool
        If True resample the data.
    param_resample_sfreq: float
        New sample rate to use.
    param_resample_npad: int or str
        Amount to pad the start and end of the data. Can be “auto” (default).
    param_resample_window: str
        Frequency-domain window to use in resampling. 
    param_resample_stim_picks: list of int or None
        Stim channels.
    param_resample_n_jobs: int
        Number of jobs to run in parallel.
    param_resample_events: 2D array, shape (n_events, 3) or None
        An optional event matrix. 
    param_resample_pad: str
        The type of padding to use. Supports all numpy.pad() mode options. Can also be “reflect_limited” (default).

    Returns
    -------
    raw_filtered: instance of mne.io.Raw
        The raw data after filtering.
    """

    raw.load_data()

    # Bandpass, lowpass, or highpass filter
    raw_filtered = raw.filter(l_freq=param_filter_l_freq, h_freq=param_filter_h_freq, 
                              picks=param_filter_picks, filter_length=param_filter_length,
                              l_trans_bandwidth=param_filter_l_trans_bandwidth,
                              h_trans_bandwidth=param_filter_h_trans_bandwidth, n_jobs=param_filter_n_jobs,
                              method=param_filter_method, iir_params=param_filter_iir_params, phase=param_filter_phase,
                              fir_window=param_filter_fir_window, fir_design=param_filter_fir_design,
                              skip_by_annotation=param_filter_skip_by_annotation, pad=param_filter_pad)

    # Notch
    if param_apply_notch is True:
        freqs = np.arange(param_notch_freqs_start, param_notch_freqs_end, param_notch_freqs_step)
        raw_filtered.notch_filter(freqs=freqs, picks=param_notch_picks,
                                  filter_length=param_notch_filter_length, notch_widths=param_notch_widths,
                                  trans_bandwidth=param_notch_trans_bandwidth, n_jobs=param_notch_n_jobs,
                                  method=param_notch_method, iir_params=param_notch_iir_parameters,
                                  mt_bandwidth=param_notch_mt_bandwidth, p_value=param_notch_p_value,
                                  phase=param_notch_phase, fir_window=param_notch_fir_window,
                                  fir_design=param_notch_fir_design, pad=param_notch_pad)

    # Resample
    if param_apply_resample is True:
        raw_filtered.resample(sfreq=param_resample_sfreq, npad=param_resample_npad, window=param_resample_window,
                              stim_picks=param_resample_stim_picks, n_jobs=param_resample_n_jobs,
                              events=param_resample_events, pad=param_resample_pad)

    # Save file
    raw.save("out_dir_temporal_filtering/meg.fif", overwrite=True)

    return raw_filtered


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
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

    # Check for None parameters for lowpass, highpass, or band pass filter
    
    # l_freq
    if config['param_filter_l_freq'] == "":
        config['param_filter_l_freq'] = None  # when App is run on Bl, no value for this parameter corresponds to ''
    
    # h_freq
    if config['param_filter_h_freq'] == "":
        config['param_filter_h_freq'] = None  # when App is run on Bl, no value for this parameter corresponds to ''

    # picks
    if config['param_filter_picks'] == "":
        config['param_filter_picks'] = None  # when App is run on Bl, no value for this parameter corresponds to ''

    # iir parameters
    if config['param_filter_iir_params'] == "":
        config['param_filter_iir_params'] = None  # when App is run on Bl, no value for this parameter corresponds to ''

    # Info message about filtering

    # Band pass filter
    if config['param_filter_l_freq'] is not None and config['param_filter_h_freq'] is not None:
        comments_about_filtering = f'Data was filtered between ' \
                                   f'{config["param_filter_l_freq"]} ' \
                                   f'and {config["param_filter_h_freq"]}Hz'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})

    # Lowpass filter
    elif config['param_filter_l_freq'] is None and config['param_filter_h_freq'] is not None:
        comments_about_filtering = f'Lowpass filter was applied at {config["param_filter_h_freq"]}Hz'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})

    # Highpass filter
    elif config['param_filter_l_freq'] is not None and config['param_filter_h_freq'] is None:
        comments_about_filtering = f'Highpass filter was applied at {config["param_filter_l_freq"]}Hz'
        dict_json_product['brainlife'].append({'type': 'info', 'msg': comments_about_filtering})

    # Raise an exception if both param_filter_l_freq and param_filter_h_freq are None
    elif config['param_filter_l_freq'] is None and config["param_filter_h_freq"] is None:
        value_error_message = f'You must specify a value for param_filter_l_freq or param_filter_h_freq, ' \
                              f"they can't both be set to None"
        # Raise exception
        raise ValueError(value_error_message)

    # Info message about notch filtering if applied
    if config['param_apply_notch'] is True:
        dict_json_product['brainlife'].append({'type': 'info', 'msg': 'Notch filter was applied.'})
        comments_notch = f"{config['param_notch_freqs_start']}Hz and its harmonics"

        # Check for None parameters 

        # picks notch
        if config['param_notch_picks'] == "":
            config['param_notch_picks'] = None  # when App is run on Bl, no value for this parameter corresponds to ''

        # notch widths
        if config['param_notch_widths'] == "":
            config['param_notch_widths'] = None  # when App is run on Bl, no value for this parameter corresponds to ''  

        # iir parameters
        if config['param_notch_iir_parameters'] == "":
            config['param_notch_iir_parameters'] = None  # when App is run on Bl, no value for this parameter corresponds to ''  

        # mt bandwidth
        if config['param_notch_mt_bandwidth'] == "":
            config['param_notch_mt_bandwidth'] = None  # when App is run on Bl, no value for this parameter corresponds to ''         

    else:
        comments_notch = 'No Notch filter was applied'

    # Info message about resampling if applied
    if config['param_apply_resample'] is True:
        if config['param_resample_sfreq'] is not None:
            dict_json_product['brainlife'].append({'type': 'info', 'msg': f'Data was resampled at '
                                                                          f'{config["param_resample_sfreq"]}. '
                                                                          f'Please bear in mind that it is generally '
                                                                          f'recommended not to epoch '
                                                                          f'downsampled data, but instead epoch '
                                                                          f'and then downsample.'})
            comments_resample_freq = f'{config["param_resample_sfreq"]}Hz'

            # Check for None parameters

            # stim picks
            if config['param_resample_stim_picks'] == "":
              config['param_resample_stim_picks'] = None  # when App is run on Bl, no value for this parameter corresponds to ''  

            if config['param_resample_events'] == "":
              config['param_resample_events'] = None  # when App is run on Bl, no value for this parameter corresponds to '' 
            
        else:
            value_error_message = f"You must specify a value for param_resample_sfreq. If you don't want to resample " \
                                  f'your data, please set param_apply_resample to False.'
            # Raise exception
            raise ValueError(value_error_message)
    else:
        comments_resample_freq = 'Data was not resampled'

    # Keep bad channels in memory
    bad_channels = raw.info['bads']

    # Define kwargs
    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  

    # Apply temporal filtering
    raw_copy = raw.copy()
    raw_filtered = temporal_filtering(raw_copy, **kwargs)
    del raw_copy

    # Success message in product.json    
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Filtering was applied successfully.'})

    # Compute SNR
    snr_before = _compute_snr(raw)
    snr_after = _compute_snr(raw_filtered)

    # Generate a report
    _generate_report(data_file, raw, raw_filtered, bad_channels, comments_about_filtering,
                     comments_notch, comments_resample_freq, snr_before, snr_after)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
