# 🚀 Quick Start Guide - AQI Predictor

## **For Streamlit Cloud Deployment**

### **⚡ 5-Minute Deploy**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Click "New app"
   - Select your repo: `your-username/AQI_Predictor`
   - Main file: `app/streamlit_app.py`
   - Click "Deploy"

3. **Add Secrets** (App Settings → Secrets):
   ```toml
   AQICN_API_KEY = "your_token"
   HOPSWORKS_API_KEY = "your_key"
   HOPSWORKS_PROJECT_NAME = "aqi_predictor99"
   CITY = "karachi"
   CITY_LAT = "24.8607"
   CITY_LON = "67.0011"
   ```

4. **Done!** Your app is live at `https://yourapp.streamlit.app/`

---

## **For Local Development**

### **Setup:**
```bash
# 1. Clone repository
git clone your-repo-url
cd AQI_Predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure .env
copy .env.example .env
# Edit .env with your API keys

# 5. Run dashboard
streamlit run app/streamlit_app.py
```

### **Open:**
http://localhost:8501

---

## **Quick Commands**

```bash
# Validate configuration
python main.py --check-config

# Test inference
python main.py --pipeline predict

# Run dashboard
streamlit run app/streamlit_app.py

# Test changes
python validate_changes.py
```

---

## **Need Help?**

- **Full deployment guide:** Read `DEPLOYMENT_GUIDE.md`
- **Technical details:** Read `STREAMLIT_INTEGRATION_SUMMARY.md`
- **Recent changes:** Read `README_CHANGES.md`

---

**That's it! You're ready to go! 🎉**
