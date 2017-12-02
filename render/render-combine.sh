#!/bin/bash

TMPDIR=/tmp/moronkai
RENDER_NAME=head_final11
TRAIN_RENDERS=1500000
TEST_RENDERS=150000

echo "Creating directories"
rm -rf ${TMPDIR}/${RENDER_NAME}*
mkdir -p ${TMPDIR}/${RENDER_NAME}_train
mkdir -p ${TMPDIR}/${RENDER_NAME}_test

echo "Extracting render parts to temp"
for f in ${WRKDIR}/renders/${RENDER_NAME}/*.tar; do
  time (tar -xf $f -C ${TMPDIR}/${RENDER_NAME}_train) &
done
wait

WC=($(find ${TMPDIR}/${RENDER_NAME}_train/ -type f | wc))
TOTAL_RENDERS=${WC[0]}
RENDERS_TO_REMOVE=$(($TOTAL_RENDERS - ($TRAIN_RENDERS + $TEST_RENDERS)))

echo "Total number of renders: ${TOTAL_RENDERS}"
echo "Renders to remove: ${RENDERS_TO_REMOVE}"

if (($RENDERS_TO_REMOVE < 0)); then
  echo "Not enough total number of renders!"
  exit 1
fi

if (($RENDERS_TO_REMOVE > 0)); then
  echo "Removing extra renders"
  time (find ${TMPDIR}/${RENDER_NAME}_train/ -type f | sort | head -${RENDERS_TO_REMOVE} | xargs rm)
fi

echo "Moving renders to test"
time (find ${TMPDIR}/${RENDER_NAME}_train/ -type f | sort | head -${TEST_RENDERS} | xargs mv -t ${TMPDIR}/${RENDER_NAME}_test)

TOTAL_TRAIN_RENDERS=$(find ${TMPDIR}/${RENDER_NAME}_train/ -type f | wc -l)
TOTAL_TEST_RENDERS=$(find ${TMPDIR}/${RENDER_NAME}_test/ -type f | wc -l)

echo "Total number of train renders: ${TOTAL_TRAIN_RENDERS}"
echo "Total number of test renders: ${TOTAL_TEST_RENDERS}"

echo "Creating tar archive"
cd ${TMPDIR}
time (tar -cf ${RENDER_NAME}.tar ${RENDER_NAME}_train ${RENDER_NAME}_test)

echo "Moving tar archive to WRKDIR"
time (mv ${TMPDIR}/${RENDER_NAME}.tar ${WRKDIR}/renders/)
