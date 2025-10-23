# Complete workflow for fixing your exoplanet detection training

# Step 1: Activate your environment (already done)
# conda activate exo-lstm-gpu

# Step 2: Copy the fixed files to your Code directory
Write-Host "Copying fixed scripts to Code directory..." -ForegroundColor Green
Copy-Item "diagnose_data.py" "C:\CS_4280_Project\Code\diagnose_data.py"
Copy-Item "train_lstm_fixed.py" "C:\CS_4280_Project\Code\train_lstm_fixed.py"

# Step 3: Navigate to Code directory
Set-Location "C:\CS_4280_Project\Code"

# Step 4: Run diagnostics to check your data
Write-Host "`nStep 1: Running diagnostics..." -ForegroundColor Cyan
python diagnose_data.py

Write-Host "`nPress Enter to continue with training (or Ctrl+C to stop)..." -ForegroundColor Yellow
Read-Host

# Step 5: Run the fixed training script
Write-Host "`nStep 2: Starting training..." -ForegroundColor Cyan
python train_lstm_fixed.py `
  --windows_dir "C:\CS_4280_Project\Code\data\windows_train" `
  --epochs 40 `
  --batch_size 256 `
  --lr 1e-4 `
  --hidden 128 `
  --layers 2 `
  --dropout 0.3 `
  --save_dir "C:\CS_4280_Project\Code\runs\lstm_fixed" `
  --amp_dtype fp16 `
  --pos_weight 3.37 `
  --num_workers 0

Write-Host "`nTraining complete!" -ForegroundColor Green
