# app-resampling

This is a draft of a future Brainlife App that resamples MEG signals using the MNE functions:  
[`mne.io.Raw.resample`](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample) or 
[`mne.Epochs.resample`](https://mne.tools/stable/generated/mne.Epochs.html?highlight=mne%20epochs#mne.Epochs.resample).

# app-resampling documentation

1) Resample MEG signals 
2) Input file is:
    * a MEG file in `.fif` format,
    * an optional fine calibration file in `.dat`,
    * an optional crosstalk compensation file in `.fif`,
    * an optional head position file in `.pos`,
    * an optional destination file in `.fif`,
    * an optional event file in `.tsv`.
3) Input parameters are:
    * param_epoched_data: `bool`, if True, the data to be resampled is epoched, else it is continuous.
    * param_resample_sfreq: `float`, new sample rate to use.
    * param_resample_npad: `int` or `str`, amount to pad the start and end of the data. Default is 'auto'.
    * param_resample_window: `str`, frequency-domain window to use in resampling. Default is `boxcar`. 
    * param_resample_stim_picks: `list of /*int` or `None`, stim channels. Default is `None`.
    * param_resample_n_jobs: `int`, number of jobs to run in parallel. Default is 1. 
    * param_resample_events: `2D array, shape (n_events, 3)`, optional, an optional event matrix. Default is `None`.
    * param_raw_pad: `str`, the type of padding to use for raw data. Default is 'reflect_limited'. 
    * param_epoch_pad: `str`, the type of padding to use for epoched data. Default is 'edge'. 

This list along with the parameters' default values correspond to the 0.22.0 version of MNE Python.  

5) Ouput files are:
    * a `.fif` MEG file after resampling,
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

