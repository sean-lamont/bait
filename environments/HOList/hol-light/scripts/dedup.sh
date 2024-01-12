#!/bin/bash
#
# Removes duplicate proofs from test and eval sets.
# Expects files named e.g. base.train, base.valid, and base.test.
#
# Usage: ./dedup.sh path/to/base path/to/output/base

set -exuo pipefail
ulimit -s 65536

TRAIN="$1.train"
TEST="$1.test"
VALID="$1.valid"

SORTED_TRAIN="${TRAIN}.s"
SORTED_TEST="${TEST}.s"
SORTED_VALID="${VALID}.s"

# Sort and remove duplicates within each file
cat $TRAIN | sort -u > $SORTED_TRAIN
cat $TEST | sort -u > $SORTED_TEST
cat $VALID | sort -u > $SORTED_VALID

# Record duplicates between training and test/valid files
awk 'a[$0]++' <(cat $SORTED_TEST) <(cat $SORTED_TRAIN) > test_train.dups
awk 'a[$0]++' <(cat $SORTED_VALID) <(cat $SORTED_TRAIN) > valid_train.dups

# Remove recorded training duplicates from test and valid sets
comm -23 $SORTED_VALID valid_train.dups > "${SORTED_VALID}.notr"
comm -23 $SORTED_TEST test_train.dups > "${SORTED_TEST}.notr"

# Record duplicates between test and valid sets
awk 'a[$0]++' <(cat "${SORTED_VALID}.notr") <(cat "${SORTED_TEST}.notr") > test_valid.dups

# Remove duplicates from valid set
comm -23 "${SORTED_TEST}.notr" test_valid.dups > "${SORTED_TEST}.notr.noval"

# Shuffle
shuf $SORTED_TRAIN > "$2.train"
shuf "${SORTED_VALID}.notr" > "$2.valid"
shuf "${SORTED_TEST}.notr.noval" > "$2.test"

# Cleanup
rm "${SORTED_TRAIN}"
rm "${SORTED_TEST}"
rm "${SORTED_VALID}"
rm *.notr
rm *.noval
rm *.dups
