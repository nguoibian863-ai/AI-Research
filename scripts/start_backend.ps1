# PowerShell script to start FastAPI backend
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Starting Local Deep Research Agent API on http://127.0.0.1:8000 ..." -ForegroundColor Green
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
