import os
import numpy as np
import argparse
import mne
import warnings
import multiprocessing


from mne_hfo import RMSDetector
from natsort import natsorted
from tabulate import tabulate

CORE_N = multiprocessing.cpu_count()

# Define verbosity levels
VERBOSITY_NONE = 0      # No output
VERBOSITY_LOW = 1       # Basic information
VERBOSITY_MEDIUM = 2    # Detailed information
VERBOSITY_HIGH = 3      # Debug level information

# Custom help message epilog
custom_epilog = """
From MNE-HFO:
Root mean square (RMS) detection algorithm (Staba Detector)

The original algorithm described in the reference, takes a sliding
window of 3 ms, computes the RMS values between 100 and 500 Hz.
Then events separated by less than 10 ms were combined into one event.
Then events not having a minimum of 6 peaks (i.e. band-pass signal
rectified above 0 V) with greater then 3 std above mean baseline
were removed. A finite impulse response (FIR) filter with a
Hamming window was used.

The script's default threshold is slightly different from original and is 5 std.

The detection band also differs slightly from the original and is 80-500 Hz.

References
[1] R. J. Staba, C. L. Wilson, A. Bragin, I. Fried, and J. Engel,
“Quantitative Analysis of High-Frequency Oscillations (80 − 500 Hz)
Recorded in Human Epileptic Hippocampus and Entorhinal Cortex,”
J. Neurophysiol., vol. 88, pp. 1743–1752, 2002.
"""

def read_edf(path, drop_non_eeg=True, drop_EEG_Prefix=False, preload=False, verbose=0):
    """
    Read an EDF file with configurable verbosity levels:
    0 (NONE) - No output
    1 (LOW) - Basic file information
    2 (MEDIUM) - Channel information and preprocessing steps
    3 (HIGH) - Detailed debug information
    """
    if verbose >= VERBOSITY_LOW:
        print(f"Reading {path}...\n")
    
    path = str(path)
    non_eeg = [
        "DC1",
        "Baseline",
        "ECG",
        "EKG",
        "EMG",
        "MD/Pic",
        "MD",
        "Pic",
        "Mic",
        "Mic-0",
        "Mic-1",
        "Mic-2",
        "Mic-3",
        "Mic-4",
        "Mic-5",
        "Motor",
        "Music",
        "Noise",
        "Picture",
        "Story",
        "ECG ECG",
        "EEG ECG",
        "Pt Mic",
        "MD Mic",
        "PT Mic",
        "Hand Motor",
        "ECG EKG",
        "EKG ECG",
        "Hand",
        "EDF Annotations",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        names = mne.io.read_raw_edf(path, preload=False, verbose=False).ch_names
        non_eeg = [x for x in non_eeg if x in names]

    if drop_non_eeg:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mne_raw = mne.io.read_raw_edf(path, preload=preload, 
                                         verbose='DEBUG' if verbose >= VERBOSITY_HIGH else False, 
                                         exclude=non_eeg)
        eeg = [x for x in mne_raw.ch_names if x.startswith("EEG ")]

        additional_non_eeg = mne_raw.ch_names
        additional_non_eeg = [x for x in additional_non_eeg if x not in eeg]

        # remove non EEG channels
        mne_raw.drop_channels(additional_non_eeg)
        dropped = additional_non_eeg + non_eeg
        if verbose >= VERBOSITY_MEDIUM:
            print("\nIgnoring non EEG channels: ")
            for i, x in enumerate(dropped):
                print(f"{i+1}: {x}")
            print("\n")

        sEEG_picks = mne.pick_types(mne_raw.info, eeg=True, exclude=[])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mne_raw = mne.io.read_raw_edf(path, preload=preload, 
                                         verbose='DEBUG' if verbose >= VERBOSITY_HIGH else False)
        sEEG_picks = np.arange(len(mne_raw.ch_names))

    if drop_EEG_Prefix:
        mne_raw.rename_channels(lambda x: x.replace("EEG ", ""))
        mne_raw.rename_channels(lambda x: x.replace(" ", ""))
        mne_raw.rename_channels(lambda x: x.replace(' ','').strip())

    chNames = np.array(mne_raw.ch_names)[sEEG_picks]

    if any(["-" in x for x in chNames]):
        print("WARNING: Channel names contain the '-' character. Bipolar namings use this character. Replacing with '~'.")

        issue_channels = [x for x in chNames if "-" in x]
        if verbose >= VERBOSITY_MEDIUM:
            print("The following channels contain '-' and which will be replaced with '~' in output: ")
            for i, x in enumerate(issue_channels):
                print(f"{i+1}: {x}")
            print("\n")
        renames = {x: x.replace("-", "~") for x in issue_channels}
        mne_raw.rename_channels(renames)
        chNames = np.array(mne_raw.ch_names)[sEEG_picks]


    primes = [x for x in chNames if "'" in x]
    non_primes = [x for x in chNames if "'" not in x]
    prime_n = len(primes)
    non_prime_n = len(non_primes)
    primes = list(natsorted(primes))
    non_primes = list(natsorted(non_primes))

    if len(non_primes) > 0:
        non_primes.extend(primes)
        chNames = non_primes
    else:
        chNames = primes
    if len(chNames) != prime_n+non_prime_n:
        raise ValueError("Something went wrong with the channel names")

    z = mne_raw.pick(chNames).reorder_channels(chNames)

    if verbose >= VERBOSITY_LOW:
        print(f"Loaded {len(z.ch_names)} channels from {path}")
        
        if verbose >= VERBOSITY_MEDIUM:
            print(f"Channels: {z.ch_names}")
            print(f"Sampling frequency: {z.info['sfreq']} Hz")
            print(f"Duration: {z.times[-1] / 60:.2f} minutes")
        print("\n")
    return z

def _strip_prefix(input_string):
    # remove the EEG prefix from a string
    if input_string.startswith("EEG "):
        return input_string[4:]
    else:
        return input_string

def rereference(mne_raw_org, rereference_scheme="none", todrop=[], verbose=0):
    """
    Rereference the EEG data with configurable verbosity levels:
    0 (NONE) - No output
    1 (LOW) - Basic rereferencing information
    2 (MEDIUM) - Channel modification details
    3 (HIGH) - Debug level information including all channel operations
    """
    if type(rereference_scheme) is type(None):
        rereference_scheme = "none"

    reference_scheme = rereference_scheme.lower()
    known_schemes = ["none", "average", "bipolar"]
    if reference_scheme not in known_schemes:
        raise ValueError(f"Unknown rereference scheme: \"{rereference_scheme}\". Known schemes are: {known_schemes}")

    if verbose >= VERBOSITY_LOW:
        print(f"Applying \"{reference_scheme}\" referencing scheme...\n")

    mne_raw = mne_raw_org.copy()
    if rereference_scheme == "none" or rereference_scheme is None:
        return mne_raw

    elif rereference_scheme == "average":
        if len(todrop) > 0:
            if verbose >= VERBOSITY_LOW:
                print(f"Dropping {len(todrop)} channels before average reference.")
            if verbose >= VERBOSITY_MEDIUM:
                print(f"Dropping channels: {todrop}\n")
            mne_raw.drop_channels(todrop)
        mne_raw.set_eeg_reference(ref_channels="average", projection=True, 
                                 verbose='DEBUG' if verbose >= VERBOSITY_HIGH else False)
        return mne_raw

    elif rereference_scheme == "bipolar":            
        chNames = mne_raw.ch_names
        chNames_stripped = [_strip_prefix(c) for c in chNames]
        to_use = np.array(natsorted(set(chNames_stripped)))

        # organize the channels if there are primes and non-primes
        primes = [x for x in to_use if "'" in x]
        non_primes = [x for x in to_use if "'" not in x]
        prime_n = len(primes)
        non_prime_n = len(non_primes)
        primes = natsorted(primes)
        non_primes = natsorted(non_primes)

        if len(non_primes) > 0:
            non_primes.extend(primes)
            to_use = non_primes
        else:
            to_use = primes
        if len(to_use) != prime_n+non_prime_n:
            raise ValueError("Something went wrong with the channel names while rereferencing!")
        to_use = np.array(to_use)

        # get electrode labels
        allLabels = []
        for x in to_use:
            if "'" in x:
                beg = x.split("'")[0]
                beg = beg + "'"
            else:
                beg = ''.join([c for c in x if not c.isdigit()])
            allLabels.append(beg)

        bipolarIdx = np.argwhere([allLabels[i] == allLabels[i + 1] for i in range(len(allLabels) - 1)])[:, 0]
        anodes = to_use[bipolarIdx]
        cathodes = to_use[bipolarIdx + 1]

        # match to stripped names to get the original names
        anodes = [chNames[chNames_stripped.index(x)] for x in anodes]
        cathodes = [chNames[chNames_stripped.index(x)] for x in cathodes]

        if verbose >= VERBOSITY_MEDIUM:
            new_labels = np.array(allLabels)[bipolarIdx]
            pairs = [f"{a}-{c}" for a,c in list(zip(anodes, cathodes))]
            print("Bipolar reference pairs:")
            print(tabulate({
                "Electrode": new_labels,
                "Anode": anodes,
                "Cathode": cathodes,
                "Bipolar Label": pairs
            }, headers="keys", tablefmt="rounded_outline"))
            print("\n")

        mne_raw = mne.set_bipolar_reference(
            mne_raw,
            anodes,
            cathodes,
            ch_name=None,
            ch_info=None,
            drop_refs=True,
            copy=True,
            verbose='DEBUG' if verbose >= VERBOSITY_HIGH else False
        )
        if verbose >= VERBOSITY_MEDIUM:
            print("\n")

        if len(todrop) > 0:
            ch_names = mne_raw.ch_names
            ch_names_stripped = [c.replace("EEG ","") for c in ch_names]
            ch_name_tuples = [tuple(c.split("-")) for c in ch_names_stripped]
            to_drop = [_strip_prefix(c) for c in todrop]

            indexs = []
            for x in to_drop:
                for i,(a,b) in enumerate(ch_name_tuples):
                    if x == a or x == b:
                        indexs.append(i)
            indexs = list(set(indexs))
            drop_electrodes = natsorted([ch_names[i] for i in indexs])
            mne_raw.drop_channels(drop_electrodes)
            if verbose >= VERBOSITY_LOW:
                print(f"Dropping {len(drop_electrodes)} bipolar channels after rereference.\n")
            if verbose >= VERBOSITY_MEDIUM:
                print(f"Dropping bipolar channels: {drop_electrodes}\n")


        if verbose >= VERBOSITY_LOW:
            print(f"{len(mne_raw.ch_names)} bipolar channels after rereference and drops.\n")

        return mne_raw


# ############
# # Main script

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Process EDF files for HFO detection using the Staba RMS method.',
    epilog=custom_epilog,
    formatter_class=argparse.RawDescriptionHelpFormatter # Preserve formatting of epilog
)
parser.add_argument("input_file", help="Path to the input EDF file.")
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3],
                    default=1, help="Verbosity level (0=None, 1=Low, 2=Medium, 3=High). Default: 1")
parser.add_argument("--dry_run", action='store_true',
                    help="Perform a dry run, printing parameters and checks without processing or saving data.")
parser.add_argument("--overwrite", action='store_true',
                    help="Overwrite existing output files if they exist.")
parser.add_argument("--hfo_band_low", type=int, default=80,
                    help="Lower frequency bound for HFO detection in Hz. Default: 80")
parser.add_argument("--hfo_band_high", type=int, default=None,
                    help="Upper frequency bound for HFO detection in Hz. If None, defaults to Nyquist frequency / 2 - 1. Default: None")
parser.add_argument("--window_size_sec", type=float, default=0.003,
                    help="Sliding window size in seconds for RMS calculation. Default: 0.003")
parser.add_argument("--overlap", type=float, default=0.25,
                    help="Fraction of window overlap for RMS calculation (0 to 1). Default: 0.25")
parser.add_argument("--threshold", type=float, default=5.0,
                    help="Threshold in standard deviations above baseline for HFO detection. Default: 5.0")
parser.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of CPU cores to use for parallel processing. -1 uses all available cores. Default: -1")
parser.add_argument("--output_path", type=str, default=None,
                    help="Directory path to save output CSV files. Defaults to the script's directory.")
parser.add_argument("--save_type", type=str, choices=["raw", "counts", "both"], default="both",
                    help="Type of output to save ('raw' HFO events, 'counts' per channel, or 'both'). Default: both")
parser.add_argument("--to_drop", nargs='+', default=[],
                    help="List of channel names to drop before counting HFOs (separated by spaces [e.g. --to_drop Fp1 Fp2 Q'1]).")
parser.add_argument("--rereference_scheme", type=str, choices=["bipolar", "average", "none"], default="bipolar",
                    help="Rereferencing scheme to apply ('bipolar', 'average', or 'none'). Default: bipolar")
parser.add_argument("--disable_notch_filter", action='store_true',
                    help="Disable the 60Hz notch filter and its harmonics.")


args = parser.parse_args()
edf_path = args.input_file
verbosity = args.verbosity
dry_run = args.dry_run
overwrite = args.overwrite
hfo_band = (args.hfo_band_low, args.hfo_band_high)
window_size_sec = args.window_size_sec
overlap = args.overlap
threshold = args.threshold
n_jobs = args.n_jobs
output_path = args.output_path
save_type = args.save_type
to_drop = args.to_drop
rereference_scheme = args.rereference_scheme # Get rereference scheme from args
apply_notch = not args.disable_notch_filter # Get notch filter flag from args


# Check if the input file exists
if not os.path.exists(edf_path):
    raise ValueError(f"Input file {edf_path} does not exist")

if save_type not in ["raw", "counts", "both"]:
    raise ValueError(f"Invalid save type: {save_type}. Must be one of: raw, counts, both. both is default.")

# Check if the output path is valid
if output_path is None:
    # get the directory of the script
    output_path = os.path.dirname(os.path.abspath(__file__))

os.makedirs(output_path, exist_ok=True)

if os.path.exists(output_path):
    if verbosity >= VERBOSITY_LOW:
        print(f"Output directory: {output_path}")
else:
    raise ValueError(f"Attempted to make output directory {output_path} but it still does not exist!")


# check if files already exist
sourcefile = os.path.basename(edf_path)
name = os.path.splitext(sourcefile)[0]
if not overwrite:
    if verbosity >= VERBOSITY_LOW:
        print(f"Checking for existing files in {output_path}...\n")
    if save_type == "both":
        if os.path.exists(os.path.join(output_path, f"{name}_hfo_raw.csv")):
            print(f"WARNING: {name}_hfo_raw.csv already exists in {output_path}.")
            print("Use --overwrite to overwrite existing files.")
            exit(1)
        if os.path.exists(os.path.join(output_path, f"{name}_hfo_counts.csv")):
            print(f"WARNING: {name}_hfo_counts.csv already exists in {output_path}.")
            print("Use --overwrite to overwrite existing files.")
            exit(1)
    elif save_type == "raw":
        if os.path.exists(os.path.join(output_path, f"{name}_hfo_raw.csv")):
            print(f"WARNING: {name}_hfo_raw.csv already exists in {output_path}.")
            print("Use --overwrite to overwrite existing files.")
            exit(1)
    elif save_type == "counts":
        if os.path.exists(os.path.join(output_path, f"{name}_hfo_counts.csv")):
            print(f"WARNING: {name}_hfo_counts.csv already exists in {output_path}.")
            print("Use --overwrite to overwrite existing files.")
            exit(1)



if dry_run:
    if verbosity <= VERBOSITY_MEDIUM:
        verbosity = VERBOSITY_MEDIUM

# check that hfo band is a tuple, if not check if it is a single value
if isinstance(hfo_band, (int, float)):
    hfo_band = (hfo_band, None)
if isinstance(hfo_band, (tuple, list)):
    if len(hfo_band) == 1:
        hfo_band = (hfo_band[0], None)
    elif len(hfo_band) != 2:
        raise ValueError("HFO band must be a tuple of two values (l_freq, h_freq)!")
    
if hfo_band[1] is not None:
    if isinstance(hfo_band[0], float) or isinstance(hfo_band[1], float):
        print("WARNING: HFO band frequencies must be integer values! Converting to int.")
        print(f"Converted HFO band")
        print(f"From: {hfo_band[0]} - {hfo_band[1]}")
        hfo_band = (int(hfo_band[0]), int(hfo_band[1]))
        print(f"To: {hfo_band[0]} - {hfo_band[1]}")
    if hfo_band[0] > hfo_band[1]:
        raise ValueError("HFO band must be in the form (l_freq, h_freq) with l_freq < h_freq!")
    
# check that window size is a float or int
if isinstance(window_size_sec, (int, float)):
    window_size_sec = float(window_size_sec)
if window_size_sec <= 0:
    raise ValueError("Window size (sec) must be a positive value > 0!")

# check that overlap is a float or int
if isinstance(overlap, (int, float)):
    overlap = float(overlap)
if overlap < 0 or overlap > 1:
    raise ValueError("Overlap must be a float value between 0 and 1 representing the fraction of overlap!")

# check that threshold is a float or int
if isinstance(threshold, (int, float)):
    threshold = float(threshold)
if threshold <= 0:
    raise ValueError("Threshold must be a positive value > 0!")

# check that n_jobs is an int
if isinstance(n_jobs, int):
    pass # Already int
else:
    raise ValueError("n_jobs must be an integer value!")
if n_jobs == -1: # Explicitly check for -1 to use all cores
    n_jobs = CORE_N
    if verbosity >= VERBOSITY_LOW: # Only print if verbosity is Low or higher
        print(f"n_jobs set to {n_jobs} (all available cores).\n")
elif n_jobs < 1:
    # Handle cases like 0 or other negative numbers if necessary, though argparse default prevents this unless manually set < -1
    if verbosity >= VERBOSITY_LOW: # Only print if verbosity is Low or higher
        print(f"WARNING: n_jobs must be -1 or a positive integer. Setting to all available cores ({CORE_N}).\n")
    n_jobs = CORE_N
elif n_jobs > CORE_N:
    n_jobs = CORE_N
    if verbosity >= VERBOSITY_LOW: # Only print if verbosity is Low or higher
        print(f"WARNING: n_jobs set to {n_jobs} (max available cores) because requested value is greater than available cores.\n")

# ensure to drop format is correct
if isinstance(to_drop, str):
    to_drop = [to_drop]
if isinstance(to_drop, list):
    if any("-" in x for x in to_drop):
        print("WARNING: The following channel names to drop contain the '-' character. ")
        print("Bipolar namings use this character. Replacing with '~'.")
        issue_channels = [x for x in to_drop if "-" in x]
        print(issue_channels)
    to_drop = [x.replace("-", "~") for x in to_drop]

# Read and preprocess the EDF file
mne_raw = read_edf(edf_path, drop_EEG_Prefix=True, preload=True, verbose=verbosity)
sf = mne_raw.info['sfreq']

if apply_notch: # Check if notch filter should be applied
    if verbosity >= VERBOSITY_LOW:
        print(f"Applying notch filter at 60Hz harmonics...")
        
    mne_raw.notch_filter(np.arange(60, int(sf // 2), 60), 
                        picks='eeg', 
                        filter_length='auto', 
                        phase='zero',
                        fir_window='hamming', 
                        fir_design='firwin', 
                        verbose=verbosity >= VERBOSITY_HIGH)


ch_names = mne_raw.ch_names
stripped_names = [_strip_prefix(c) for c in ch_names]
stripped_drops = [_strip_prefix(c) for c in to_drop]

# update to_drop to replace "-" with "~"
to_drop = [x.replace("-", "~") for x in to_drop]

droppable = [x for x in stripped_drops if x in stripped_names]
not_droppable = [x for x in stripped_drops if x not in stripped_names]

if len(not_droppable) > 0:
    if verbosity >= VERBOSITY_LOW:
        print("\n")
        print("The following channels will be dropped from the data:")
        for i, x in enumerate(droppable):
            print(f"{i+1}: {x}")

    print("\n")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("The following channels were not found in the data and cannot not be dropped:")
    for i, x in enumerate(not_droppable):
        print(f"{i+1}: {x}")
    print("Please check the channel names in the data. ")
    print("(either set a higher verbosity level or set dry_run=True for information about channels in data)")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\n")

mne_raw = rereference(mne_raw, rereference_scheme=rereference_scheme, verbose=verbosity, todrop=droppable) # Pass scheme to function

window_size_samples = int(window_size_sec * sf)
if verbosity >= VERBOSITY_LOW:
    print(f"HFO detection parameters:")
    print(f'-----------------------------------------')
    print(f"Window size: {window_size_samples} samples ({window_size_sec} seconds)")
    print(f"Overlap: {overlap * 100:.2f}%")
    print(f"Threshold: {threshold} standard deviations")
    print("\n")

# Configure HFO detector
detector_kwargs = {
    'filter_band': hfo_band,  # (l_freq, h_freq)
    'threshold': threshold,    # Number of st. deviations
    'win_size': window_size_samples,  # Sliding window size in samples
    'overlap': overlap,        # Fraction of window overlap [0, 1]
    'hfo_name': "hfo",
    "n_jobs": n_jobs,
    "verbose": verbosity >= VERBOSITY_HIGH  # Only show detector debug info at high verbosity
}

# Check if band is above Nyquist frequency
nyquist_limit = int(sf // 2) - 1
if hfo_band[1] is None:
    hfo_band = (hfo_band[0], nyquist_limit)
    if verbosity >= VERBOSITY_LOW:
        print(f"Upper HFO band limit defaulted to Nyquist frequency / 2 - 1 ({nyquist_limit} Hz)\n") # Improved message
    detector_kwargs['filter_band'] = hfo_band

if max(hfo_band) > nyquist_limit: # Use the calculated limit for comparison
    print("ERROR:")
    print(f"Defined HFO band ({hfo_band[0]}-{hfo_band[1]} Hz) is at or above Nyquist frequency ({sf//2} Hz)")
    print(f"Upper band limit must be set strictly below Nyquist frequency. (Max upper limit: {nyquist_limit} Hz)")
    print("Please check the band settings.")
    print("Exiting...")
    exit(1)

hfo_band = (int(hfo_band[0]), int(hfo_band[1])) # ensure band is int
print(f"Detecting HFOs in band: {hfo_band[0]} - {hfo_band[1]}\n")
if dry_run:
    print("Dry run complete. No data output.")
    print("To run the script, set dry_run=False.")
    exit(0)


if verbosity >= VERBOSITY_MEDIUM:
    print(f"Initializing RMS detector with {n_jobs} cores...\n")

rms_detector = RMSDetector(**detector_kwargs)

if verbosity >= VERBOSITY_LOW:
    print("Fitting RMS detector to data...\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    rms_detector.fit(mne_raw)

output_df = rms_detector.hfo_df.copy()

output_df.drop(columns=["label"], inplace=True)

output_df.rename(columns={"onset": "onset (sec)", "duration": "duration (sec)", "sample": "onset (sample)"}, inplace=True)
output_df["onset (sample)"] = output_df["onset (sample)"].astype(int)
channels = output_df["channels"]
channel_a = [x.split("-")[0] for x in channels]
channel_b = [x.split("-")[1] for x in channels]
output_df["channel a"] = channel_a
output_df["channel b"] = channel_b

sourcefile = os.path.basename(edf_path)
output_df["source file"] = sourcefile

duration_sec = output_df["duration (sec)"]
duration_samples = duration_sec * sf
output_df["duration (samples)"] = duration_samples.astype(int)

# reorder columns
output_df = output_df[["source file", 
                       "channels", 
                       "channel a", 
                       "channel b", 
                       "onset (sec)", 
                       "onset (sample)",
                       "duration (sec)",
                       "duration (samples)"]]

df_counts = output_df.groupby("channels").size().reset_index(name="hfo count")
channels = df_counts["channels"]
channel_a = [x.split("-")[0] for x in channels]
channel_b = [x.split("-")[1] for x in channels]
df_counts["channel a"] = channel_a
df_counts["channel b"] = channel_b
df_counts["source file"] = sourcefile

r = mne_raw.times[-1]
df_counts['recording length (sec)'] = r


df_counts["hfo rpm"] = (df_counts["hfo count"] / r) * 60

df_counts = df_counts[["source file", 
                       "channels", 
                       "channel a", 
                       "channel b", 
                       "hfo count",
                       "recording length (sec)",
                       "hfo rpm"]]

# Save the output DataFrame to a CSV file
sourcefile = os.path.basename(edf_path)
name = os.path.splitext(sourcefile)[0]

if save_type == "both":
    output_df.to_csv(os.path.join(output_path, f"{name}_hfo_raw.csv"), index=False)
    df_counts.to_csv(os.path.join(output_path, f"{name}_hfo_counts.csv"), index=False)
    if verbosity >= VERBOSITY_LOW:
        print(f"Saved both raw hfo and hfo counts data to {output_path}")
elif save_type == "raw":
    output_df.to_csv(os.path.join(output_path, f"{name}_hfo_raw.csv"), index=False)
    if verbosity >= VERBOSITY_LOW:
        print(f"Saved raw hfo data to {output_path}")
elif save_type == "counts":
    df_counts.to_csv(os.path.join(output_path, f"{name}_hfo_counts.csv"), index=False)
    if verbosity >= VERBOSITY_LOW:
        print(f"Saved hfo counts data to {output_path}")

if verbosity >= VERBOSITY_LOW:
    print("HFO detection complete!")


