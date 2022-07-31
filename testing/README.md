# Testing

We have divided our tests into two categories, `functional_tests` and `unit_tests`. In each
folder, you will find a `t_utils` file with some helper functions, and a `t_template` file,
which provides a testing template.

## Running in GitHub

Checkout `ivadomed/.github/workflows/run_tests.yml` to see how tests are run on pull requests.

## Running Locally

1. Install dependencies
```
cd ivadomed  # root of the repo
pip install -e .[dev]
```

2. Download the required dataset(s) using the `ivadomed` command line tools:
```
cd ivadomed  # root of the repo
ivadomed_download_data -d data_testing -o data_testing  # for unit tests
ivadomed_download_data -d data_functional_testing -o data_functional_testing  # for functional tests
```
3. To run all tests:
```
pytest
```
or, to run specific tests:
```
pytest testing/functional_tests/
pytest testing/unit_tests/
pytest testing/functional_tests/test_example.py
```

## Wiki

You can read more about our testing here: https://github.com/ivadomed/ivadomed/wiki/Tests
