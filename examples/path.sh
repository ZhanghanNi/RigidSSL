#!/bin/bash

export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PERTURB_DATA_DIR="/path/to/RigidSSL_Perturb_data"
export MD_DATA_DIR="/path/to/RigidSSL_MD_data"
export OUTPUT_BASE_DIR="$PROJECT_DIR/output"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
