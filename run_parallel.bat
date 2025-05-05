@echo off
echo ======================================
echo Mapping AI Startups - Parallel Runner
echo ======================================
echo.
echo 1. First checking API keys...
python check_api_keys.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo API key check failed! Please fix your API keys in the .env file.
    echo.
    pause
    exit /b 1
)

echo.
echo 2. Starting parallel processing...
python parallel_processor.py

echo.
echo Processing completed!
echo.
pause 