@echo off
echo ====================================================================
echo Comparing Balanced vs Unbalanced Models
echo ====================================================================
echo.

python compare_balanced_vs_unbalanced.py

echo.
echo ====================================================================
echo Now testing balanced model on 100 real planets...
echo ====================================================================
echo.

python inference_cluster_model.py ^
  --model_path "runs/bilstm_cluster_balanced/best.pt" ^
  --windows_dir "data/windows_planet_test" ^
  --output_file "reports/balanced_planet_predictions.csv"

echo.
echo ====================================================================
echo COMPLETE! Results saved
echo ====================================================================
echo.
echo Files created:
echo   - reports/balanced_planet_predictions.csv (new)
echo   - reports/optimized_planet_predictions.csv (existing)
echo.
echo Compare these files to see which model gives fewer false alarms!
echo.
pause
