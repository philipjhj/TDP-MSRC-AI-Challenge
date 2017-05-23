@ECHO OFF
setlocal
set PYTHONPATH=%cd%\..\pig_chase;%cd%;%cd%\..;%cd%\..\..\..\ProjectMalmo\Python_Examples
python %1
endlocal