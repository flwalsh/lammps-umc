#!/usr/bin/env bash
#flux: --job-name fit
#flux: --queue pbatch
#flux: --nodes 4
#flux: --time-limit 2h
#flux: --bank XXXXX

export HSA_XNACK=1

source /path/to/venv/grace/bin/activate

flux run -N4 -n16 -c24 -g1 /path/to/peacemaker.py -r input.yaml
