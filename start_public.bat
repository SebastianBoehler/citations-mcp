@echo off
setlocal enabledelayedexpansion

echo [citations-mcp] Starting Research Citations MCP Server for Windows
echo.

REM Configuration
set "LOCAL_PORT=%MCP_SSE_LOCAL_PORT%"
if "%LOCAL_PORT%"=="" set "LOCAL_PORT=8000"

set "NGROK_BIN=%MCP_SSE_NGROK_BIN%"
if "%NGROK_BIN%"=="" set "NGROK_BIN=ngrok"

set "NGROK_AUTH_TOKEN=%NGROK_AUTHTOKEN%"
if "%NGROK_AUTH_TOKEN%"=="" set "NGROK_AUTH_TOKEN=%MCP_SSE_NGROK_AUTHTOKEN%"

REM Check for required commands and install if missing
echo [citations-mcp] Checking requirements...

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [citations-mcp] Python not found. Installing Python...
    call :install_python
)

REM Check UV
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [citations-mcp] UV not found. Installing UV...
    call :install_uv
)

REM Check ngrok
where %NGROK_BIN% >nul 2>nul
if %errorlevel% neq 0 (
    echo [citations-mcp] Ngrok not found. Installing Ngrok...
    call :install_ngrok
)

REM Check configuration
echo [citations-mcp] Checking configuration...
call :check_config

REM Create temporary directory for logs
set "TMP_DIR=%TEMP%\citations-mcp-%RANDOM%"
mkdir "%TMP_DIR%" 2>nul
set "NGROK_LOG=%TMP_DIR%\ngrok.log"

REM Start the MCP SSE server in background
echo [citations-mcp] Starting Research Citations MCP server on port %LOCAL_PORT%
start /B cmd /c "uv run uvicorn src.main:app --host 0.0.0.0 --port %LOCAL_PORT% %*"

REM Give the server time to start
timeout /t 3 /nobreak >nul

REM Start ngrok tunnel
echo [citations-mcp] Starting ngrok tunnel...

if "%NGROK_AUTH_TOKEN%"=="" (
    start /B %NGROK_BIN% http --log=stdout --log-format=json %LOCAL_PORT% >"%NGROK_LOG%" 2>&1
) else (
    start /B %NGROK_BIN% http --log=stdout --log-format=json --authtoken %NGROK_AUTH_TOKEN% %LOCAL_PORT% >"%NGROK_LOG%" 2>&1
)

REM Wait for ngrok to report the public URL using PowerShell
echo [citations-mcp] Waiting for ngrok tunnel to start...
powershell -Command "& {$timeout = 30; $start = Get-Date; while (((Get-Date) - $start).TotalSeconds -lt $timeout) { if (Test-Path '%NGROK_LOG%') { $content = Get-Content '%NGROK_LOG%' -Raw; $lines = $content -split '`n'; foreach ($line in $lines) { if ($line -match 'started tunnel') { try { $json = $line | ConvertFrom-Json; if ($json.url) { Write-Output $json.url; exit 0; } } catch { } } } } Start-Sleep -Milliseconds 500 } Write-Error 'Timeout waiting for ngrok URL'; exit 1 }" >"%TMP_DIR%\url.txt" 2>&1

if %errorlevel% neq 0 (
    echo [citations-mcp] error: ngrok tunnel did not start within 30 seconds
    echo [citations-mcp] ngrok log:
    type "%NGROK_LOG%"
    rmdir /S /Q "%TMP_DIR%" >nul 2>nul
    pause
    exit /b 1
)

REM Read the URL from the temporary file
set /p public_url=<"%TMP_DIR%\url.txt"

if "%public_url%"=="" (
    echo [citations-mcp] error: Could not extract ngrok URL
    rmdir /S /Q "%TMP_DIR%" >nul 2>nul
    pause
    exit /b 1
)
REM Display connection information
echo.
echo [citations-mcp] ✅ Server is running!
echo [citations-mcp] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo [citations-mcp] Public URL: %public_url%
echo [citations-mcp] MCP SSE Endpoint: %public_url%/mcp/sse
echo [citations-mcp] Health Check: %public_url%/health
echo [citations-mcp] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo [citations-mcp] Add this to ChatGPT MCP config:
echo [citations-mcp]   URL: %public_url%/mcp/sse
echo.
echo [citations-mcp] Press Ctrl+C to stop the server
echo.

REM Keep the window open and wait for user to stop
:wait_server
timeout /t 60 /nobreak >nul
goto wait_server

REM Configuration checking function
:check_config
echo [citations-mcp] Checking .env file configuration...

REM Check if .env file exists
if not exist ".env" (
    echo [citations-mcp] Error: .env file not found.
    echo [citations-mcp] Please create a .env file with your OpenAI API key and papers directory.
    pause
    exit /b 1
)

REM Check OpenAI API key
findstr /B "OPENAI_API_KEY=" .env >nul 2>&1
if %errorlevel% neq 0 (
    echo [citations-mcp] Error: OPENAI_API_KEY not found in .env file.
    pause
    exit /b 1
)

REM Check papers directory
findstr /B "PAPERS_DIRECTORY=" .env >nul 2>&1
if %errorlevel% neq 0 (
    echo [citations-mcp] Error: PAPERS_DIRECTORY not found in .env file.
    pause
    exit /b 1
)

echo [citations-mcp] Configuration loaded from .env file!
goto :eof

REM Installation functions
:install_python
echo [citations-mcp] Installing Python...
echo [citations-mcp] Checking if winget is available...
where winget >nul 2>nul
if %errorlevel% equ 0 (
    echo [citations-mcp] Using winget to install Python...
    winget install Python.Python.3 --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 (
        echo [citations-mcp] Python installed successfully!
        echo [citations-mcp] Please close this window and run the script again.
        pause
        exit /b 0
    ) else (
        echo [citations-mcp] Winget installation failed. Trying manual download...
    )
)

echo [citations-mcp] Opening Python download page in your browser...
echo [citations-mcp] Please:
echo [citations-mcp] 1. Download Python from the page that opens
echo [citations-mcp] 2. Run the installer
echo [citations-mcp] 3. IMPORTANT: Check "Add Python to PATH"
echo [citations-mcp] 4. Click "Install Now"
echo [citations-mcp] 5. After installation, close this window and run the script again
start https://www.python.org/downloads/
pause
exit /b 1

:install_uv
echo [citations-mcp] Installing UV...
echo [citations-mcp] Checking if winget is available...
where winget >nul 2>nul
if %errorlevel% equ 0 (
    echo [citations-mcp] Using winget to install UV...
    winget install astral-sh.uv --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 (
        echo [citations-mcp] UV installed successfully!
        goto :eof
    ) else (
        echo [citations-mcp] Winget installation failed. Trying PowerShell method...
    )
)

echo [citations-mcp] Installing UV using PowerShell...
powershell -Command "& { try { irm https://astral.sh/uv/install.ps1 | iex; Write-Output 'UV installed successfully!' } catch { Write-Error 'Installation failed'; exit 1 } }"
if %errorlevel% equ 0 (
    echo [citations-mcp] UV installed successfully!
    echo [citations-mcp] Please close this window and run the script again.
    pause
    exit /b 0
) else (
    echo [citations-mcp] PowerShell installation failed.
    echo [citations-mcp] Please visit https://docs.astral.sh/uv/getting-started/installation/
    echo [citations-mcp] and install UV manually, then run this script again.
    start https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

:install_ngrok
echo [citations-mcp] Installing Ngrok...
set "NGROK_DIR=%USERPROFILE%\ngrok"
set "NGROK_EXE=%NGROK_DIR%\ngrok.exe"

if exist "%NGROK_EXE%" (
    echo [citations-mcp] Ngrok already installed locally.
    set "NGROK_BIN=%NGROK_EXE%"
    goto :eof
)

echo [citations-mcp] Creating ngrok directory...
mkdir "%NGROK_DIR%" 2>nul

echo [citations-mcp] Downloading ngrok...
powershell -Command "& { try { Invoke-WebRequest -Uri 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip' -OutFile '%NGROK_DIR%\ngrok.zip'; Expand-Archive -Path '%NGROK_DIR%\ngrok.zip' -DestinationPath '%NGROK_DIR%' -Force; Remove-Item '%NGROK_DIR%\ngrok.zip'; Write-Output 'Ngrok downloaded successfully!' } catch { Write-Error 'Download failed'; exit 1 } }"

if %errorlevel% equ 0 (
    echo [citations-mcp] Ngrok installed successfully to %NGROK_DIR%
    set "NGROK_BIN=%NGROK_EXE%"
    goto :eof
) else (
    echo [citations-mcp] Automatic download failed.
    echo [citations-mcp] Opening ngrok download page in your browser...
    echo [citations-mcp] Please:
    echo [citations-mcp] 1. Download the Windows zip file from the page that opens
    echo [citations-mcp] 2. Unzip it to C:\ngrok\ or any folder you like
    echo [citations-mcp] 3. Run this script again
    start https://ngrok.com/download
    pause
    exit /b 1
)

REM Cleanup on exit
if exist "%TMP_DIR%" (
    rmdir /S /Q "%TMP_DIR%" >nul 2>nul
)
pause
