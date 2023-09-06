#!/bin/bash
cd hol/HOL/
poly < tools/smart-configure.sml 
bin/build cleanAll
bin/build

