# HFO Detection Script (Staba RMS Method)

This script processes EEG data from EDF files to detect High-Frequency Oscillations (HFOs) using the Root Mean Square (RMS) detection algorithm described by Staba et al. (2002). It outputs the detected HFO events and counts per channel into CSV files.

## Features

*   Uses the MNE_HFO library with the Staba RMS HFO detection method.
*   Reads EDF files and automatically handles common non-EEG channels.
*   Applies bipolar referencing.
*   Applies a notch filter to remove power line noise (60Hz and harmonics).
*   Highly configurable detection parameters via command-line arguments (frequency band, threshold, window size, overlap).
*   Variable verbosity levels for detailed output.
*   Option to perform a dry run to check parameters without processing data.
*   Outputs results in easy-to-use CSV format (raw events and counts per channel).

## Installation

It is recommended to use a virtual environment (like conda or venv) to manage dependencies.

1.  **Create and activate a conda environment (Recommended):**
    install the latest version of conda if you don't have it already. You can download it from [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions). I recommend using the Miniconda version, which is smaller and faster to install, the full version is not necessary for this script.
    ```bash
    conda create -n hfos python=3.12
    conda activate hfos
    ```
    *(To deactivate later: `conda deactivate`)*

2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal, providing the path to the input EDF file.

**Basic Usage:**

```bash
python HFOs_edf_to_csv.py /path/to/your_data.edf
```

**Command-Line Arguments:**

Use the `--help` flag to see all available options and their descriptions:

```bash
python HFOs_edf_to_csv.py --help
```

This will display detailed information about each argument, including defaults and choices, as well as the algorithm description below.

**Example with Custom Parameters:**

```bash
python HFOs_edf_to_csv.py 098_Baseline.EDF --hfo_band_low 100 --hfo_band_high 300 --threshold 3.0 --save_type counts --output_path ./results --overwrite -v 2
```

This command:
*   Processes `098_Baseline.EDF`.
*   Sets the HFO detection band to 100-300 Hz.
*   Sets the detection threshold to 3.0 standard deviations.
*   Saves only the HFO counts per channel.
*   Specifies the output directory as `./results`.
*   Overwrites existing output files.
*   Sets the verbosity level to Medium (detailed output).

## Algorithm Details (Staba Detector)

From MNE-HFO:

Root mean square (RMS) detection algorithm (Staba Detector)

The original algorithm described in the reference, takes a sliding
window of 3 ms, computes the RMS values between 100 and 500 Hz.
Then events separated by less than 10 ms were combined into one event.
Then events not having a minimum of 6 peaks (i.e. band-pass signal
rectified above 0 V) with greater then 3 std above mean baseline
were removed. A finite impulse response (FIR) filter with a
Hamming window was used.

*   **Note:** The script's default threshold is slightly different from original and is 5 std.
*   **Note:** The detection band also differs slightly from the original and is 80-500 Hz (default lower bound is 80 Hz, upper bound defaults to Nyquist/2 - 1 if not specified).

## Output Files

By default (`--save_type both`), the script generates two CSV files in the specified output directory (or the script's directory if not specified):

1.  **`<input_filename>_hfo_raw.csv`**: Contains details for each detected HFO event.
    *   `source file`: Original EDF filename.
    *   `channels`: Bipolar channel name (e.g., 'A1-A2').
    *   `channel a`: Anode channel name.
    *   `channel b`: Cathode channel name.
    *   `onset (sec)`: Event onset time in seconds from the start of the recording.
    *   `onset (sample)`: Event onset time as sample index.
    *   `duration (sec)`: Event duration in seconds.
    *   `duration (samples)`: Event duration in samples.

2.  **`<input_filename>_hfo_counts.csv`**: Contains the total HFO count and rate for each bipolar channel.
    *   `source file`: Original EDF filename.
    *   `channels`: Bipolar channel name.
    *   `channel a`: Anode channel name.
    *   `channel b`: Cathode channel name.
    *   `hfo count`: Total number of HFOs detected on this channel.
    *   `recording length (sec)`: Total duration of the recording in seconds.
    *   `hfo rpm`: HFO rate in events per minute (count / duration_minutes).

## References

[1] R. J. Staba, C. L. Wilson, A. Bragin, I. Fried, and J. Engel,
“Quantitative Analysis of High-Frequency Oscillations (80 − 500 Hz)
Recorded in Human Epileptic Hippocampus and Entorhinal Cortex,”
J. Neurophysiol., vol. 88, pp. 1743–1752, 2002.







