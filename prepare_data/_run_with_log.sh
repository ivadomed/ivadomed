#!/bin/bash
# Wrapper to be able to output stdout and stderr in text file

TASK=$1
SUBJECT=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

cd ${PATH_DATA}
${TASK} $SUBJECT $PATH_OUTPUT $PATH_QC $PATH_LOG 2>&1 | tee ${PATH_LOG}/${SUBJECT}.log ; test ${PIPESTATUS[0]} -eq 0
if [ ! $? -eq 0 ]; then
  mv ${PATH_LOG}/${SUBJECT}.log ${PATH_LOG}/err.${SUBJECT}.log
fi
