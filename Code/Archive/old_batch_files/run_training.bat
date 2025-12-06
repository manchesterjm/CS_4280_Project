@echo off
echo Starting Autonomous Training Pipeline...
echo.

call conda activate exo-lstm-gpu

echo Activated environment: exo-lstm-gpu
echo.

cd /d C:\CS_4280_Project\Code

echo Running training with window sizes: 2048, 1024, 4096
echo This will take several hours. Results will be saved to runs\sector1_experiments\
echo.

python autonomous_training_pipeline.py --window_sizes 2048 1024 4096

echo.
echo Training complete!
