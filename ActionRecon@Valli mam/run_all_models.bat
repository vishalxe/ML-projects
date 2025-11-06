

echo [5/5] Aggregating Results...
echo ----------------------------------------
python aggregate_results.py
if %errorlevel% neq 0 (
    echo ERROR: Results aggregation failed
    pause
    exit /b 1
)
echo Results aggregation completed successfully!
echo.

echo ========================================
echo ALL MODELS COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Generated files:
echo - Individual model results (JSON, CSV, PNG)
echo - Performance summary tables
echo - Comparative visualizations
echo - Statistical analysis
echo - Comprehensive report
echo.
echo Check the following files for results:
echo - comprehensive_results_report.md
echo - performance_summary.csv
echo - Various PNG visualization files
echo.
echo Results are ready for journal publication!
echo.
pause
