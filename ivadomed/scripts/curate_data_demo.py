"""
The data are structure in a folder such as:
dataset/
├── 20220309_demo_mouse_07_0001
│         ├── image.png
│         ├── mask_axon.png
│         ├── mask_axonmyelin.png
│         ├── mask_myelin.png
│         └── pixel_size_in_micrometer.txt
├── 20220309_demo_mouse_07_0002
│         ├── image.png
│         ├── mask.png
│         ├── mask_seg-axon-manual.png
│         ├── mask_seg-myelin-manual.png
│         └── pixel_size_in_micrometer.txt
└── 20220309_demo_mouse_07_0003
          ├── image.png
          ├── mask.png
          ├── mask_seg-axon-manual.png
          ├── mask_seg-myelin-manual.png
          └── pixel_size_in_micrometer.txt

"""

# Define various dependencies

import glob
import os
import shutil
import json
import argparse
import subprocess
import csv
from textwrap import dedent

## Dictionary for images
images = {
    "image.png": "_TEM.png"
}

## Dictionary for derivatives
der = {
    "mask.png": "_TEM_seg-axonmyelin-manual.png",
    "mask_seg-axon-manual.png": "_TEM_seg-axon-manual.png",
    "mask_seg-myelin-manual.png": "_TEM_seg-myelin-manual.png"
}

# Define function to get the input and output path for data
def get_parameters():
    parser = argparse.ArgumentParser(description='This script is curating the demo dataset to BIDS')
    # Define input path
    parser.add_argument("-d", "--data",
                        help="Path to folder containing the dataset to be curated",
                        required=True)
    # Define output path
    parser.add_argument("-o", "--outputdata",
                        help="Path to output folder",
                        required=True,
                        )
    arguments = parser.parse_args()
    return arguments

# Define function to create json sidecar
def create_json_sidecar(output_data, sub_id):

    # Path for each subject  destination bids folder
    path_folder_sub_id_bids = os.path.join(output_data, sub_id, 'microscopy')

    # Create filename for json sidecar
    item_out = sub_id + "_TEM.json"

    # Define json sidecar content
    data_json = {"PixelSize": [0.00236, 0.00236],
                 "FieldOfView": [8.88, 5.39],
                 "BodyPart": "BRAIN",
                 "BodyPartDetails": "splenium",
                 "SampleFixation": "2% paraformaldehyde, 2.5% glutaraldehyde",
                 "Environment": "exvivo"
                 }

    # Write content to file
    with open(os.path.join(path_folder_sub_id_bids, item_out), 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

# Define main function of the script
def main(root_data, output_data):

    # Curate the contents of the dataset to keep only folders and sort them
    contents_ds = [subdir for subdir in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, subdir))]
    contents_ds.sort()

    # Loop across contents of each subdirectory
    for subdir in contents_ds:

        # Define subject id
        sub_id = "sub-demoMouse" + subdir.split('_')[3]

        # Define sample id
        sample_id = subdir.split('_')[4]

        # Get the path of each subdirectory
        path_subdir = os.path.join(root_data, subdir)
        # Get the contents of each subdirectory
        contents_subdir = os.listdir(path_subdir)

        # Define final bids subject id
        sub_bids_full = sub_id + "_sample-" + sample_id

        # Loop across the contents of each subdirectory
        for file in contents_subdir:

            # Get the path of each file
            path_file_in = os.path.join(path_subdir, file)

            # Check if the filename corresponds to the one in the images dictionary
            if file in images:

                # Most files go into the subject's data folder
                path_sub_id_dir_out = os.path.join(output_data, sub_id, 'micr')

                # Define the output file path
                path_file_out = os.path.join(path_sub_id_dir_out, sub_bids_full + images[file])

            # Check if the filename corresponds to the one in the derivatives dictionary
            elif file in der:

                # Derivatives go somewhere else
                path_sub_id_dir_out = os.path.join(output_data, 'derivatives', 'labels', sub_id, 'micr')

                # Define the output file path
                path_file_out = os.path.join(path_sub_id_dir_out, sub_bids_full + der[file])
            else:
                # not a file we recognize
                continue

            # Create output subdirecotries and copy files to output
            os.makedirs(os.path.dirname(path_file_out), exist_ok=True)
            shutil.copyfile(path_file_in, path_file_out)

    # Generate subject list
    sub_list = sorted(d for d in os.listdir(output_data) if d.startswith("sub-"))

    # Now that everything is curated, fill in the metadata
    for sub_id in sub_list:
        create_json_sidecar(output_data, sub_id)

    # Create participants.tsv and samples.tsv
    with open(output_data + '/samples.tsv', 'w') as samples, \
            open(output_data + '/participants.tsv', 'w') as participants:
        tsv_writer_samples = csv.writer(samples, delimiter='\t', lineterminator='\n')
        tsv_writer_samples.writerow(["sample_id", "participant_id", "sample_type"])
        tsv_writer_participants = csv.writer(participants, delimiter='\t', lineterminator='\n')
        tsv_writer_participants.writerow(["participant_id", "species"])
        for subject in sub_list:
            row_sub = []
            row_sub.append(subject)
            row_sub.append('mus musculus')
            tsv_writer_participants.writerow(row_sub)
            subject_samples = sorted(glob.glob(os.path.join(output_data, subject, 'micr', '*.png')))
            for file_sample in subject_samples:
                row_sub_samples = []
                row_sub_samples.append(os.path.basename(file_sample).split('_')[1])
                row_sub_samples.append(subject)
                row_sub_samples.append('tissue')
                tsv_writer_samples.writerow(row_sub_samples)

    # Create dataset_description.json
    dataset_description = {"Name": "demo dataset",
                           "BIDSVersion": "1.7.0"
                           }

    with open(output_data + '/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description, json_file, indent=4)

    # Create dataset_description.json for derivatives/labels
    dataset_description_derivatives = {"Name": "demo dataset labels",
                                       "BIDSVersion": "1.7.0",
                                       "GeneratedBy": [{"Name": "demo dataset labels pipeline name"}]
                                       }

    with open(output_data + '/derivatives/labels/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description_derivatives, json_file, indent=4)

    # Create participants.json
    data_json = {
        "participant_id": {
            "Description": "Unique participant ID"
        },
        "species": {
            "Description": "Binomial species name from the NCBI Taxonomy (https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi)"
        }
    }

    with open(output_data + '/participants.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # Create samples.json
    data_json = {
        "sample_id": {
            "Description": "Sample ID"
        },
        "participant_id": {
            "Description": "Participant ID from whom tissue samples have been acquired"
        },
        "sample_type": {
            "Description": "Type of sample from ENCODE Biosample Type (https://www.encodeproject.org/profiles/biosample_type)"
        }
    }

    with open(output_data + '/samples.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

        # Create README
        with open(output_data + '/README', 'w') as readme_file:
            print(dedent("""\
    - Generate on 2022-03-09 
    - Created for demo purposes"""), file=readme_file)

# Call main function
if __name__ == "__main__":
    args = get_parameters()
    main(args.data, args.outputdata)