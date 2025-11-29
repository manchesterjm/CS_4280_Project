@echo off
echo ======================================
echo SECTOR 1 TRAINING
echo ======================================
echo.

call conda activate exo-lstm-gpu
echo Environment activated.
echo.

cd /d C:\CS_4280_Project\Code

echo Starting training...
python train_sector1.py

echo.
echo Training script completed.
