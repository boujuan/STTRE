#!/bin/bash
# Save this as debug_wrapper.sh and make it executable (chmod +x debug_wrapper.sh)
python -m debugpy --listen 5678 --wait-for-client $1
