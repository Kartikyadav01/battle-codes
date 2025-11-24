@echo off
REM Quick test script for Docker deployment (Windows)

echo 🐳 Testing Docker Build...
docker build -t rajasthan-tourism-app:test .

if %errorlevel% equ 0 (
    echo ✅ Docker build successful!
    echo.
    echo To run the container:
    echo docker run -d -p 8501:8501 --name test-app rajasthan-tourism-app:test
    echo.
    echo To test locally:
    echo docker run -p 8501:8501 -e WEATHERAPI_KEY=test -e GROQ_API_KEY=test rajasthan-tourism-app:test
) else (
    echo ❌ Docker build failed!
    exit /b 1
)

