# PowerShell script to verify and start Ollama model service
$ErrorActionPreference = "Stop"

Write-Host "Checking Ollama status..." -ForegroundColor Cyan
try {
    $models = ollama list
    Write-Host "Available Ollama models:" -ForegroundColor Green
    $models
} catch {
    Write-Warning "Ollama is not running. Please launch Ollama Desktop or run 'ollama serve'."
}
