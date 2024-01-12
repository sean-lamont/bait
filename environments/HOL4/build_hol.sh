#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
git clone https://github.com/HOL-Theorem-Prover/HOL.git
cd HOL/
git checkout kananaskis-14
poly < tools/smart-configure.sml
bin/build cleanAll
bin/build







