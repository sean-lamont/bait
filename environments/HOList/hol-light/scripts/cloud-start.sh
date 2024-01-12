#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
: ${BUILD_SCRIPT:=${SCRIPT_DIR}/cloud-build.sh}
readonly INSTANCE_NAME=${1:-${USER}-hol}
: ${ZONE:=us-central1-b}

echo "Provisioning GCloud Instance '${INSTANCE_NAME}' on zone ${ZONE}..."
time gcloud compute instances create \
  --metadata-from-file startup-script=${BUILD_SCRIPT} \
  --zone=${ZONE} \
  ${INSTANCE_NAME}

echo
echo "Provisioned GCloud Instance '${INSTANCE_NAME}'.  Please give it a few 
minutes to complete its startup script.  You can...
  * ssh into the machine with 'gcloud compute ssh ${INSTANCE_NAME}'
  * monitor the progress of its startup script with 'tail -f /var/log/syslog'
  * rerun the startup script with
    'sudo google_metadata_script_runner --script-type startup'
  * access hol-light under '/build/hol-light-master'
  * delete the cluster with 'gcloud compute instances delete ${INSTANCE_NAME}'
  * some permission modification / copying might be necessary to run hol-light
    as an unpriviliged user.
"
