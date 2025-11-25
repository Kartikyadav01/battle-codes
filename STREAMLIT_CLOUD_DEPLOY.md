# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository

Make sure your repository has:
- ✅ `app.py` in the root directory
- ✅ `requirements.txt` with all dependencies
- ✅ All data files in `data/` directory
- ✅ All model files in `models/` directory
- ✅ All result files in `results/` directory

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - You'll see a form with these fields:

3. **Fill in the Deployment Form**

   **Option A: Standard Form (New Interface)**
   - **Repository**: Select from dropdown (e.g., `username/rajasthan-tourism-prediction`)
   - **Branch**: Select branch (usually `main`)
   - **Main file path**: Enter `app.py`
   - Click "Deploy!"

   **Option B: If Asked for GitHub URL (Alternative Interface)**
   - **Repository URL**: `https://github.com/username/repository`
   - **Main file URL**: `https://github.com/username/repository/blob/main/app.py`
   - Or if the form has a single field asking for a GitHub URL to a .py file:
     ```
     https://github.com/username/repository/blob/main/app.py
     ```
   - Replace:
     - `username` → Your GitHub username
     - `repository` → Your repository name
     - `main` → Your branch name (could be `master`)

### 4. Configure Secrets (API Keys)

After deployment:

1. Go to your app's settings (click the "⋮" menu → "Settings")
2. Click "Secrets" in the left sidebar
3. Add your API keys in TOML format:

```toml
WEATHERAPI_KEY = "your_weather_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
```

4. Click "Save"
5. The app will automatically redeploy with the new secrets

### 5. Access Your App

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

## Common Issues & Solutions

### Issue: "The field needs to contain a Github URL pointing to a .py file"

**Solution**: If you see this error, the form is expecting a full GitHub URL. Use this format:

```
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/app.py
```

**Example**:
```
https://github.com/johndoe/rajasthan-tourism-prediction/blob/main/app.py
```

### Issue: "File not found" or "Module not found"

**Solutions**:
- Ensure `app.py` is in the root directory
- Check that `requirements.txt` includes all dependencies
- Verify all data/model files are committed to git

### Issue: "API key not found"

**Solutions**:
- Add keys in Streamlit Cloud Secrets (Settings → Secrets)
- Use the exact key names: `WEATHERAPI_KEY` and `GROQ_API_KEY`
- Restart the app after adding secrets

### Issue: "Out of memory"

**Solutions**:
- Streamlit Cloud free tier has 1GB memory limit
- Consider optimizing model loading
- Upgrade to a paid tier if needed

## File Structure Requirements

Your repository should look like this:

```
your-repo/
├── app.py              ← Main application file (REQUIRED)
├── requirements.txt     ← Dependencies (REQUIRED)
├── data/
│   ├── processed/
│   └── raw/
├── models/              ← All .pkl and .h5 files
├── results/             ← CSV files for results
└── .streamlit/
    └── config.toml      ← Optional, for custom config
```

## Important Notes

1. **File Size Limit**: Streamlit Cloud has a 1GB limit per app
2. **Memory Limit**: Free tier has 1GB RAM
3. **Auto-Deploy**: App redeploys automatically on every git push
4. **HTTPS**: Automatically enabled
5. **Custom Domain**: Available on paid tiers

## Verification Checklist

Before deploying, verify:

- [ ] `app.py` exists in root directory
- [ ] `requirements.txt` is complete and up-to-date
- [ ] All data files are committed (check `git status`)
- [ ] All model files are committed
- [ ] `.env` file is NOT committed (use Secrets instead)
- [ ] Repository is public (or you have Streamlit Cloud access to private repos)
- [ ] Main branch is set correctly

## After Deployment

1. **Test the app**: Visit your app URL
2. **Check logs**: View deployment logs in Streamlit Cloud dashboard
3. **Monitor usage**: Check analytics in app settings
4. **Update secrets**: Add API keys in Settings → Secrets

## Support

If you encounter issues:
1. Check deployment logs in Streamlit Cloud
2. Verify all files are committed to git
3. Test locally first: `streamlit run app.py`
4. Review error messages in the app

---

**Quick Reference**: 
- Streamlit Cloud: https://share.streamlit.io
- Documentation: https://docs.streamlit.io/streamlit-community-cloud
- Your app URL format: `https://YOUR_APP_NAME.streamlit.app`

