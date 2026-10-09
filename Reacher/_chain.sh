#!/bin/bash
# Wait for the in-flight act_rot queue to clear, then run the overnight 2x2.
cd /home/yli113/Adapt/Reacher
while pgrep -f "_run_act.sh" > /dev/null 2>&1; do sleep 60; done
sleep 10
exec ./_run_overnight.sh
