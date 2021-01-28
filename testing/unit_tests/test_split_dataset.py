import os
import csv
import json
import pytest
import numpy as np
from ivadomed.loader import utils as imed_loader_utils
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir,  __tmp_dir__

PATH_DATA = os.path.join(__tmp_dir__, 'bids')
LOG_PATH = os.path.join(__tmp_dir__, 'log')
N = 200
N_CENTERS = 5


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [0],
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

    train, val, test = imed_loader_utils.get_subdatasets_subjects_list(split_params, PATH_DATA,
                                                                       LOG_PATH)
    return train, val, test, patient_mapping


@pytest.mark.parametrize('split_params', [{
        "fname_split": None,
        "random_seed": 6,
        "center_test": [0],
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
def test_per_patient_2(split_params):
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
        "center_test": [0],
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


def create_tsvfile():
    # Create bids path
    if not os.path.isdir(PATH_DATA):
        os.mkdir(PATH_DATA)

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

    with open(os.path.join(PATH_DATA, "participants.tsv"), 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "disability", "institution_id"])
        for item in sorted(participants):
            tsv_writer.writerow(item)

    return patient_mapping


def create_jsonfile():
    """Create dataset_description.json."""

    dataset_description = {}
    dataset_description[u'Name'] = 'Test'
    dataset_description[u'BIDSVersion'] = '1.2.1'

    with open(os.path.join(PATH_DATA, "dataset_description.json"), 'w') as outfile:
        outfile.write(json.dumps(dataset_description, indent=2, sort_keys=True))
        outfile.close()


def teardown_function():
    remove_tmp_dir()
