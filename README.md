# 🕌 Rajasthan Tourist Footfall Prediction

A comprehensive machine learning application for predicting tourist footfall in major Rajasthan cities using Random Forest, XGBoost, and LSTM models.

## 🚀 Features

- **Interactive Streamlit Dashboard** with professional React-like UI
- **Multiple ML Models**: Random Forest, XGBoost, and LSTM
- **Real-time Weather Integration** via WeatherAPI.com
- **AI-Powered Insights** using Groq API
- **Advanced Analytics**: Trend analysis, volatility tracking, feature importance
- **Scenario Forecasting**: Test different weather and event scenarios
- **Comprehensive Visualizations**: Interactive charts, heatmaps, radar plots

## 📋 Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RajasthanTourismPrediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   WEATHERAPI_KEY=your_weather_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker directly

```bash
# Build image
docker build -t rajasthan-tourism-app .

# Run container
docker run -d \
  -p 8501:8501 \
  -e WEATHERAPI_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  rajasthan-tourism-app
```

## ☁️ Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository and branch
   - Set Main file: `app.py`
   - Click "Deploy!"

3. **Configure Secrets**
   - In app settings → Secrets, add:
     ```toml
     WEATHERAPI_KEY = "your_key"
     GROQ_API_KEY = "your_key"
     ```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

## 📁 Project Structure

```
RajasthanTourismPrediction/
├── app.py                 # Main Streamlit application
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed datasets
├── models/               # Trained ML models
├── results/              # Model evaluation results
├── scripts/              # Data processing and training scripts
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── DEPLOYMENT.md        # Deployment guide
```

## 🎯 Usage

1. **Data Overview Tab**: Explore historical data, trends, and city statistics
2. **Model Performance Tab**: Compare model metrics, view residuals, feature importance
3. **Scenario Forecasting Tab**: Create custom scenarios and generate predictions
4. **Insights Studio Tab**: Deep dive into analytics and AI-powered insights

## 🔑 API Keys

The app requires API keys for enhanced features:

- **WeatherAPI.com**: For live weather data
  - Get your key at: https://www.weatherapi.com/
  
- **Groq**: For AI-powered insights
  - Get your key at: https://console.groq.com/

You can configure these keys:
- Via `.env` file (local development)
- Via Streamlit Cloud Secrets (cloud deployment)
- Via sidebar in the app (runtime configuration)

## 📊 Models

- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting model
- **LSTM**: Deep learning sequential model

## 🛡️ Security

- Never commit API keys to version control
- Use environment variables or secrets management
- `.env` file is gitignored by default

## 📝 License

This project is for educational/research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit, TensorFlow, XGBoost, and Scikit-learn**

