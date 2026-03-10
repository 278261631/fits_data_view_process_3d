@echo off
cd /d "%~dp0\.."
python "std_process\recommended_pipeline_console.py" -i "D:\your\fits_input" -o "D:\your\fits_output" --box 48 --clip-sigma 3.0 --denoise-sigma 2.0 --mix-alpha 0.7 --asinh-gain 8.0 --overwrite

