#!/bin/bash
# Download common models

python3 -c "from utils.google_utils import *;
attempt_download('weights/yolov5x.pt')"
