#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd hol-light || exit
sudo docker build -t='holist' .
sudo docker run -d -p 2000:2000 --name=holist holist
