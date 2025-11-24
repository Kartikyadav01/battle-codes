# Deployment Guide for Rajasthan Tourism Prediction App

This guide covers deploying the Streamlit application on Streamlit Cloud and using Docker.

## 🚀 Option 1: Streamlit Cloud Deployment (Recommended)

Streamlit Cloud is the easiest way to deploy Streamlit apps. It doesn't require Docker.

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set Main file path: `app.py`
   - Click "Deploy!"

3. **Configure Secrets (API Keys)**
   - In Streamlit Cloud, go to your app settings
   - Click "Secrets" in the sidebar
   - Add your API keys:
     ```toml
     WEATHERAPI_KEY = "your_weather_api_key"
     GROQ_API_KEY = "your_groq_api_key"
     ```
   - Save and the app will automatically redeploy

### Important Notes for Streamlit Cloud
- ✅ No Docker needed - Streamlit Cloud handles everything
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Auto-deploys on git push
- ⚠️ File size limit: 1GB per app
- ⚠️ Memory limit: 1GB on free tier

---

## 🐳 Option 2: Docker Deployment

Use Docker if you want to deploy on your own infrastructure (AWS, GCP, Azure, etc.)

### Prerequisites
- Docker installed
- Docker Compose (optional, for easier management)

### Build and Run

#### Using Docker Compose (Recommended)

1. **Create `.env` file** (if not exists):
   ```bash
   WEATHERAPI_KEY=your_weather_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

2. **Build and run**:
   ```bash
   docker-compose up -d
   ```

3. **Access the app**: Open `http://localhost:8501`

4. **View logs**:
   ```bash
   docker-compose logs -f
   ```

5. **Stop the app**:
   ```bash
   docker-compose down
   ```

#### Using Docker directly

1. **Build the image**:
   ```bash
   docker build -t rajasthan-tourism-app .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     -p 8501:8501 \
     -e WEATHERAPI_KEY=your_weather_api_key \
     -e GROQ_API_KEY=your_groq_api_key \
     --name rajasthan-tourism-app \
     rajasthan-tourism-app
   ```

3. **Access the app**: Open `http://localhost:8501`

4. **View logs**:
   ```bash
   docker logs -f rajasthan-tourism-app
   ```

5. **Stop the container**:
   ```bash
   docker stop rajasthan-tourism-app
   docker rm rajasthan-tourism-app
   ```

### Deploy to Cloud Platforms

#### AWS (EC2/ECS)
1. Build and push to ECR
2. Deploy using ECS or EC2 with Docker

#### Google Cloud Platform
1. Build using Cloud Build
2. Deploy to Cloud Run or GKE

#### Azure
1. Build using Azure Container Registry
2. Deploy to Azure Container Instances or AKS

#### DigitalOcean
1. Build and push to DigitalOcean Container Registry
2. Deploy to App Platform or Droplet

---

## 📋 Pre-Deployment Checklist

- [ ] All dependencies listed in `requirements.txt`
- [ ] API keys configured (via secrets or environment variables)
- [ ] All data files present in `data/` directory
- [ ] All model files present in `models/` directory
- [ ] All result files present in `results/` directory
- [ ] `.env` file created (for Docker) or secrets configured (for Streamlit Cloud)
- [ ] Tested locally before deployment

---

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**
   - Ensure all packages are in `requirements.txt`
   - Check Python version compatibility (3.11)

2. **File not found errors**
   - Verify all data/model files are included in the deployment
   - Check file paths are relative, not absolute

3. **API key errors**
   - Verify keys are set correctly in secrets/env vars
   - Check key names match exactly (case-sensitive)

4. **Memory issues**
   - Streamlit Cloud free tier has 1GB limit
   - Consider optimizing model loading or upgrading tier
   - For Docker, increase container memory limits

5. **Port issues**
   - Ensure port 8501 is exposed and not blocked
   - Check firewall settings

---

## 📊 Monitoring

### Streamlit Cloud
- Built-in analytics dashboard
- View usage stats in app settings

### Docker
- Monitor logs: `docker logs -f rajasthan-tourism-app`
- Check health: `docker ps` (should show healthy status)
- Resource usage: `docker stats rajasthan-tourism-app`

---

## 🔐 Security Best Practices

1. **Never commit API keys** to git
2. **Use environment variables** or secrets management
3. **Keep dependencies updated** for security patches
4. **Use HTTPS** in production (Streamlit Cloud provides this automatically)
5. **Limit resource access** in cloud deployments

---

## 📝 Notes

- Streamlit Cloud is recommended for quick deployment
- Docker is better for custom infrastructure or enterprise deployments
- Both methods support the same functionality
- API keys can be configured later via the sidebar in the app

---

## 🆘 Support

If you encounter issues:
1. Check the logs (Streamlit Cloud dashboard or Docker logs)
2. Verify all files are present
3. Test API keys separately using `test_apis.py`
4. Review the error messages in the app

