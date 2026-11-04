#!/bin/bash
D="/Users/yugandharreddygogireddy/Documents/Alma Models /audit"
( for s in 1 2 3 4; do "$D/run_ssm.sh" direct $s; echo "DONE direct $s" >> "$D/ssm_lanes.out"; done ) &
( for s in 1 2 3 4; do "$D/run_ssm.sh" none $s; echo "DONE none $s" >> "$D/ssm_lanes.out"; done ) &
wait; echo ALLDONE >> "$D/ssm_lanes.out"
