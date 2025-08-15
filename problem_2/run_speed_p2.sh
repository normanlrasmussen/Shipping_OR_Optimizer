#!/bin/bash
# Prevent sleep while running the simulation

echo "Starting simulation and inhibiting sleep..."

systemd-inhibit --why="Running overnight simulation" --what=sleep \
    python3 speed_test_p2.py \

echo "Simulation finished. Sleep is now allowed again."