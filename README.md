# app-resampling

This is a draft of a future Brainlife App that filters MEG signals using the MNE functions: 
[`mne.io.Raw.filter`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter), 
[`mne.io.Raw.notch_filter`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter), 
[`mne.io.Raw.resample`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample).

# app-temporal-filtering documentation

1) Filter MEG signals
2) First, apply a bandpass, highpass, or lowpass filter, then optionally apply a notch filter and resample the data  
3) Input file is:
    * a MEG file in `.fif` format,
4) Input parameters are:
    * filter_l_freq: `float`, optional, for FIR filters, the lower pass-band edge; for IIR filters, the lower cutoff frequency. If `None` the data are only low-passed.  
    * filter_h_freq: `float`, optional, for FIR filters, the upper pass-band edge; for IIR filters, the upper cutoff frequency. If `None` the data are only high-passed.
    * filter_picks: `str`or `list`, optional, channels to include. Default is `None`.
    * filter_length: `str`, length of the FIR filter to use in human-readable time units. Default is `auto`. 
    * filter_l_trans_bandwidth: `float` or `str`, width of the transition band at the low cut-off frequency in Hz. Default is `auto`.
    * filter_h_trans_bandwidth: `float` or `str`, width of the transition band at the high cut-off frequency in Hz. Default is `auto`.
    * filter_n_jobs: `int`, number of jobs to run in parallel. Default is 1. 
    * filter_method: `str`, 'fir' will use overlap-add FIR filtering, 'iir' will use IIR forward-backward filtering. Default is 'fir'.
    * filter_iir_params: `dict`, optional, dictionary of parameters to use for IIR filtering. To know how to define the dictionary go 
        [there](https://mne.tools/stable/generated/mne.filter.construct_iir_filter.html#mne.filter.construct_iir_filter). Default is `None`.
    * filter_phase: `str`, phase of the filter. Default is 'zero'.
    * filter_fir_window: `str`, the window to use in FIR design. Default is 'hamming'.
    * filter_fir_design: `str`. Default is `firwin`.
    * filter_skip_by_annotation: `str` or `list of str`, if a string (or list of str), any annotation segment that begins with the given string will not be included in
        filtering, and segments on either side of the given excluded annotated segment will be filtered separately. Default is `["edge", bad_acq_skip"]`.
    * filter_pad: `str`, the type of padding to use. Default is 'reflect_limited'.
    * apply_notch: `bool`, if `True`, apply a notch filter. Default is `True`.
    * notch_freqs_start: `int`, frequency to notch filter in Hz. Default is 50.
    * notch_freqs_end: `int`, end of the interval (in Hz) of the power lines harmonics to notch filter (this value is excluded). Default is 251.  
    * notch_freqs_step: `int`, The step in Hz to filter power lines harmonics between param_notch_freqs_start and param_notch_freqs_end. Default is 50.
    * notch_picks: `str`or `list`, optional, channels to include. Default is `None`. 
    * notch_filter_length: `str`, length of the FIR filter to use in human-readable time units. Default is `auto`. 
    * notch_widths: `float`, optional, width of the stop band in Hz. Default is `None`.
    * notch_trans_bandwidth: `float`, width of the transition band in Hz. Default is 1.0.
    * notch_n_jobs: `int`, number of jobs to run in parallel. Default is 1.
    * notch_method: `str`, 'fir' will use overlap-add FIR filtering, 'iir' will use IIR forward-backward filtering. Default is 'fir'.
    * notch_iir_params: `dict`, optional, dictionary of parameters to use for IIR filtering. To know how to define the dictionary go 
        [there](https://mne.tools/stable/generated/mne.filter.construct_iir_filter.html#mne.filter.construct_iir_filter). Default is `None`. 
    * notch_mt_bandwidth: `float`, optional, the bandwidth of the multitaper windowing function in Hz. Default is `None`.
    * notch_p_value: `float`, p-value to use in F-test thresholding to determine significant sinusoidal components. Default is 0.05.
    * notch_phase: `str`, phase of the filter, only used if method='fir'. Default is 'zero'.
    * notch_fir_window: `str`, the window to use in FIR design. Default is 'hamming'.
    * notch_fir_design: `str`. Default is 'firwin'.
    * notch_pad: `str`, the type of padding to use. Default is 'reflect_limited'.
    * apply_resample: `bool`, if True resample the data. Default is `True`.
    * param_resample_sfreq: `float`, new sample rate to use.
    * param_resample_npad: `int` or `str`, amount to pad the start and end of the data. Default is 'auto'.
    * param_resample_window: `str`, frequency-domain window to use in resampling. Default is `boxcar`. 
    * param_resample_stim_picks: `list of /*int` or `None`, stim channels. Default is `None`.
    * param_resample_n_jobs: `int`, number of jobs to run in parallel. Default is 1. 
    * param_resample_events: `2D array, shape (n_events, 3)`, optional, an optional event matrix. Default is `None`.
    * param_resample_pad: `str`, the type of padding to use. Default is 'reflect_limited'. 

This list along with the parameters' default values correspond to the 0.22.0 version of MNE Python.  

5) Ouput files are:
    * a `.fif` MEG file after filtering,
    * an `.html` report containing figures.

### Authors
- [Aurore Bussalb](aurore.bussalb@icm-institute.org)

### Contributors
- [Aurore Bussalb](aurore.bussalb@icm-institute.org)
- [Maximilien Chaumon](maximilien.chaumon@icm-institute.org)

### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications. Copy and past the following lines into your repository when using this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations
1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Running the App 

### On Brainlife.io

This App has not yet been registered in Brainlife.io.

### Running Locally (on your machine)

1. git clone this repo
2. Inside the cloned directory, create `config.json` with something like the following content with paths to your input 
   files and values of the input parameters (see `config.json.example`).

```json
{
  "fif": "rest1-raw.fif"
}
```

3. Launch the App by executing `main`

```bash
./main
```

## Output

The output files are a MEG file in `.fif` format and an `.html` report.

