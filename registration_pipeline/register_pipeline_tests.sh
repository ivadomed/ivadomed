# Tests for register_pipeline.sh

# pipeline call on individual subject
bash register_pipeline.sh "sub-barcelona05"

# Batch call
# sct_run_batch \
# -jobs 1 \
# -path-data "/home/GRAMES.POLYMTL.CA/p114001/data_nvme_p114001/data-multi-subject-hemis" \
# -path-output "./" \
# -script register_pipeline.sh


