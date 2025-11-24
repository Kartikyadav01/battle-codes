#!/bin/bash
# Quick test script for Docker deployment

echo "🐳 Testing Docker Build..."
docker build -t rajasthan-tourism-app:test .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    echo ""
    echo "To run the container:"
    echo "docker run -d -p 8501:8501 --name test-app rajasthan-tourism-app:test"
    echo ""
    echo "To test locally:"
    echo "docker run -p 8501:8501 -e WEATHERAPI_KEY=test -e GROQ_API_KEY=test rajasthan-tourism-app:test"
else
    echo "❌ Docker build failed!"
    exit 1
fi

