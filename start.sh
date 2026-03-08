#!/bin/bash
cd /home/dbest/.openclaw/workspace/home_value
source .venv/bin/activate
exec uvicorn backend.main:app --host 0.0.0.0 --port 8551 --reload >> /tmp/home_value.log 2>&1
