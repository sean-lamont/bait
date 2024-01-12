#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd hol-light || exit
sudo docker build -f Dockerfile_check_proofs --ulimit stack=1000000000 --tag check_proofs .
