@echo off
cd /d "%~dp0\.."
python "std_process\recommended_pipeline_console.py" -i "D:/github/test_flow_data/data/gy1/GY1_K024-6_NoFilter_60S_Bin2_UTC20260303_162922_-29.9C_.fit" -o "D:/github/test_flow_data/processed" --box 48 --clip-sigma 3.0 --median-ksize 3 --denoise-sigma 2.0 --mix-alpha 0.7 --overwrite

