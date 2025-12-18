@echo off
REM Training Visualization for Screen Recording
REM Run this in PowerShell and record the window

echo Starting training visualization...
echo Press any key when ready to start recording...
pause

conda activate exo-lstm-gpu
python simulate_training_visual.py

echo.
echo Recording complete! Press any key to exit...
pause
