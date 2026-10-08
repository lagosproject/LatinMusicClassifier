@echo off
setlocal
:: Add conda environment paths to PATH so that DLLs like sndfile.dll can be loaded correctly
set PATH=%~dp0env;%~dp0env\Library\bin;%~dp0env\Scripts;%PATH%
python "%~dp0scripts\generate_dataset.py" --skip_incomplete %*
endlocal
