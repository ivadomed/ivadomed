import os
import csv
import json
import shutil
import pytest
import numpy as np

from ivadomed.loader import utils as imed_loader_utils


BIDS_PATH = 'bids'
LOG_PATH = 'log'
N = 200
N_CENTERS = 5


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": ['0'],
        "method": "per_center",
        "train_fraction": 0.6,
        "test_fraction": 0.2
    }, {
        "fname_split": None,
        "random_seed": 6,
        "center_test": [],
        "method": "per_center",
        "train_fraction": 0.75,
        "test_fraction": 0.25
    }])
def load_dataset(split_params):
    patient_mapping = create_tsvfile()
    create_jsonfile()

    # Create log path
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    train, val, test = imed_loader_utils.get_subdatasets_subjects_list(split_params, BIDS_PATH, LOG_PATH)
    return train, val, test, patient_mapping


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": ['0'],
        "method": "per_center",
        "train_fraction": 0.6,
        "test_fraction": 0.2
    }])
def test_per_center_testcenter_0(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    # Verify split proportion
    assert len(train) == round(0.6 * (N - len(test)))

    # Verify there is only the test center selected
    for sub in test:
        assert patient_mapping[sub]['center'] == '0'


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [],
        "method": "per_center",
        "train_fraction": 0.2,
        "test_fraction": 0.4
    }])
def test_per_center_without_testcenter(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    test_centers = set()
    for sub in test:
        test_centers.add(patient_mapping[sub]['center'])

    training_centers = set()
    for sub in train:
        training_centers.add(patient_mapping[sub]['center'])

    # Verify the test center proportion
    assert len(test_centers) == round(N_CENTERS * 0.4)

    # Verify test and training centers are fully different
    for train_center in training_centers:
        assert train_center not in test_centers


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [],
        "method": "per_patient",
        "train_fraction": 0.45,
        "test_fraction": 0.35
    }])
def test_per_patient(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.45), atol=1)
    assert np.isclose(len(test), round(N * 0.35), atol=1)


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [],
        "method": "per_patient",
        "train_fraction": 0.6,
        "test_fraction": 0
    }])
def test_per_patient(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.6), atol=1)
    assert np.isclose(len(val), round(N * 0.4), atol=1)
    assert np.isclose(len(test), 0, atol=1)


def check_balance(train, val, test, patient_mapping):
    for dataset in [train, val, test]:
        disability_count = {'0': 0, '1': 0, '2': 0}
        for sub in dataset:
            disability_count[patient_mapping[sub]['disability']] += 1

        assert np.isclose(disability_count['0'], disability_count['1'], atol=1)
        assert np.isclose(disability_count['1'], disability_count['2'], atol=1)
        assert np.isclose(disability_count['0'], disability_count['2'], atol=1)


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [],
        "balance": "disability",
        "method": "per_patient",
        "train_fraction": 0.45,
        "test_fraction": 0.35
    }])
def test_per_patient_balance(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.45), atol=1)
    assert np.isclose(len(test), round(N * 0.35), atol=1)
    check_balance(train, val, test, patient_mapping)


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": ['0'],
        "balance": "disability",
        "method": "per_center",
        "train_fraction": 0.4,
        "test_fraction": 0.2
    }])
def test_per_center_balance(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    # Verify split proportion
    assert np.isclose(len(train), round(0.4 * (N - len(test))), atol=1)

    # Verify there is only the test center selected
    for sub in test:
        assert patient_mapping[sub]['center'] == '0'

    check_balance(train, val, test, patient_mapping)
    delete_test_folders()


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
        # 3 different disabilities: 0, 1, or 2
        disability_id = str(participant_id % 3)
        row_participants.append(disability_id)
        # N_CENTERS different centers: 0, 1, ..., or N_CENTERS
        center_id = str(participant_id % N_CENTERS)
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
