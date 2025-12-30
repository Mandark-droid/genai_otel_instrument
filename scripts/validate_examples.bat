@echo off
REM Validate All Examples Script (Windows)
REM
REM This script runs all PII, Toxicity, and Bias detection examples
REM and validates that they execute without errors.
REM
REM Usage:
REM   scripts\validate_examples.bat [endpoint]
REM
REM Example:
REM   scripts\validate_examples.bat http://192.168.206.128:55681
REM   scripts\validate_examples.bat  REM Uses default localhost:4318

setlocal enabledelayedexpansion

REM Configuration
set "ENDPOINT=%~1"
if "%ENDPOINT%"=="" set "ENDPOINT=http://localhost:4318"
set "OTEL_EXPORTER_OTLP_ENDPOINT=%ENDPOINT%"

REM Counters
set TOTAL=0
set PASSED=0
set FAILED=0
set SKIPPED=0

echo ==========================================================================
echo GenAI OTEL Examples Validation
echo ==========================================================================
echo OTEL Endpoint: %ENDPOINT%
echo ==========================================================================
echo.

echo === PII Detection Examples ===
if exist "examples\pii_detection\" (
    for %%f in (examples\pii_detection\*.py) do (
        set /a TOTAL+=1
        echo [!TOTAL!] Testing %%~nf ...

        REM Skip env_var_config.py
        if "%%~nf"=="env_var_config" (
            echo   SKIPPED (requires env vars)
            set /a SKIPPED+=1
        ) else (
            python "%%f" >nul 2>&1
            if !errorlevel! equ 0 (
                echo   PASSED
                set /a PASSED+=1
            ) else (
                echo   FAILED (exit code: !errorlevel!)
                set /a FAILED+=1
            )
        )
    )
) else (
    echo ERROR: examples\pii_detection directory not found
    exit /b 1
)
echo.

echo === Toxicity Detection Examples ===
if exist "examples\toxicity_detection\" (
    for %%f in (examples\toxicity_detection\*.py) do (
        set /a TOTAL+=1
        echo [!TOTAL!] Testing %%~nf ...

        REM Skip env_var_config.py and perspective_api.py
        if "%%~nf"=="env_var_config" (
            echo   SKIPPED (requires env vars)
            set /a SKIPPED+=1
        ) else if "%%~nf"=="perspective_api" (
            if "%PERSPECTIVE_API_KEY%"=="" (
                echo   SKIPPED (requires PERSPECTIVE_API_KEY)
                set /a SKIPPED+=1
            ) else (
                python "%%f" >nul 2>&1
                if !errorlevel! equ 0 (
                    echo   PASSED
                    set /a PASSED+=1
                ) else (
                    echo   FAILED (exit code: !errorlevel!)
                    set /a FAILED+=1
                )
            )
        ) else (
            python "%%f" >nul 2>&1
            if !errorlevel! equ 0 (
                echo   PASSED
                set /a PASSED+=1
            ) else (
                echo   FAILED (exit code: !errorlevel!)
                set /a FAILED+=1
            )
        )
    )
) else (
    echo ERROR: examples\toxicity_detection directory not found
    exit /b 1
)
echo.

echo === Bias Detection Examples ===
if exist "examples\bias_detection\" (
    for %%f in (examples\bias_detection\*.py) do (
        set /a TOTAL+=1
        echo [!TOTAL!] Testing %%~nf ...
        python "%%f" >nul 2>&1
        if !errorlevel! equ 0 (
            echo   PASSED
            set /a PASSED+=1
        ) else (
            echo   FAILED (exit code: !errorlevel!)
            set /a FAILED+=1
        )
    )
) else (
    echo WARNING: examples\bias_detection directory not found
)
echo.

echo ==========================================================================
echo VALIDATION SUMMARY
echo ==========================================================================
echo Total Examples:   !TOTAL!
echo Passed:           !PASSED!
echo Failed:           !FAILED!
echo Skipped:          !SKIPPED!
echo ==========================================================================
echo.

if !FAILED! gtr 0 (
    echo Validation FAILED
    exit /b 1
) else (
    echo Validation PASSED
    exit /b 0
)
