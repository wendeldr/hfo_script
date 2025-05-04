



# create a conda environment for HFOS
conda create -n hfos python=3.12


# activate the environment
conda activate hfos

## to deactivate the environment
conda deactivate

# install the required packages
pip install -r requirements.txt



# run the script with the --help flag to see the available options
python HFOs_edf_to_csv.py --help


# run the script to calculate HFOs from the EDF file and save results to CSV
python HFOs_edf_to_csv.py /path/to/edf_file.edf





