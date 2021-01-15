# Testing

We have divided our tests into two categories, `functional_tests` and `unit_tests`. In each
folder, you will find a `t_utils` file with some helper functions, and a `t_template` file,
which provides a testing template.

## Running in GitHub

Checkout `ivadomed/.github/workflows/run_tests.yml` to see how tests are run on pull requests.

## Running Locally

1. Download the required dataset(s) using the `ivadomed` command line tools:
```
cd ivadomed  # root of the repo
ivadomed_download_data -d data_testing -o data_testing  # for unit tests
ivadomed_download_data -d data_functional_testing -o data_functional_testing  # for functional tests
```
2. Use `pytest` to run:
```
cd ivadomed
pytest
```
or
```
cd ivadomed
pytest testing/functional_tests/
pytest testing/unit_tests/
pytets testing/functional_tests/test_example.py
```

## Wiki

You can read more about our testing here: https://github.com/ivadomed/ivadomed/wiki/Tests
