Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Menjalankan Sistem GeoSeismik BMKG (Streamlit Web App)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
& "C:\Anaconda\envs\py312\python.exe" -m streamlit run app.py
