import os
import csv
import json
import shutil
import pytest

from ivadomed.loader import utils as imed_loader_utils


BIDS_PATH = 'bids'
LOG_PATH = 'log'
N = 10


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": ['0'],
        "method": "per_center",
        "train_fraction": 0.6,
        "test_fraction": 0.2
    }])
def test_per_center_split(split_params):
    patient_mapping = create_tsvfile()
    create_jsonfile()

    # Create log path
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    train, val, test = imed_loader_utils.get_subdatasets_subjects_list(split_params, BIDS_PATH, LOG_PATH)
    assert len(test) == 0.2 * N
    assert len(train) == 0.6 * N
    assert len(val) == 0.2 * N
    for sub in test:
        assert patient_mapping[sub]['center'] == '0'


def create_tsvfile():
    # Create bids path
    if not os.path.isdir(BIDS_PATH):
        os.mkdir(BIDS_PATH)

    patient_mapping = {}

    # Create participants.tsv with n participants
    participants = []
    for participant_id in range(N):
        row_participants = []
        patient_id = 'sub-00' + str(participant_id)
        row_participants.append(patient_id)
        # 3 different disabilities: 0, 1 or 2
        disability_id = str(N % 3)
        row_participants.append(disability_id)
        # 4 different centers: 0, 1, 2 or 3
        center_id = str(N % 4)
        row_participants.append(center_id)
        patient_mapping[patient_id] = {}
        patient_mapping[patient_id]['disability'] = disability_id
        patient_mapping[patient_id]['center'] = center_id
        participants.append(row_participants)

    # # Save participants.tsv
    with open(os.path.join(BIDS_PATH, "participants.tsv"), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "disability", "institution_id"])
        for item in sorted(participants):
            tsv_writer.writerow(item)

    return patient_mapping


def create_jsonfile():
    #Create dataset_description.json
    dataset_description = {}
    dataset_description[u'Name'] = 'Test'
    dataset_description[u'BIDSVersion'] = '1.2.1'

    # Save dataset_description.json
    with open(os.path.join(BIDS_PATH, "dataset_description.json"), 'w') as outfile:
        outfile.write(json.dumps(dataset_description, indent=2, sort_keys=True))
        outfile.close()


def delete_test_folders():
    shutil.rmtree(BIDS_PATH)
    shutil.rmtree(LOG_PATH)
