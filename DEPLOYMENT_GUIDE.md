# 🚀 Streamlit Cloud Deployment Guide

## **Prerequisites**

Before deploying, ensure you have:
- ✅ GitHub repository (public or private with Streamlit auth)
- ✅ AQICN API token (https://aqicn.org/data-platform/token/)
- ✅ Hopsworks account with trained model (https://app.hopsworks.ai/)
- ✅ All code pushed to GitHub

---

## **Step 1: Prepare Your Repository**

### **1.1 Verify Required Files**

Make sure these files exist in your repo:

```
AQI_Predictor/
├── app/
│   └── streamlit_app.py         # ✅ Main dashboard
├── src/
│   ├── inference.py             # ✅ Prediction engine
│   ├── utils.py                 # ✅ Helper functions
│   └── config.py                # ✅ Configuration
├── .streamlit/
│   ├── config.toml              # ✅ Theme & settings
│   └── secrets.toml.example     # ✅ Secrets template
├── requirements.txt             # ✅ Python dependencies
├── packages.txt                 # ✅ System dependencies
└── README.md                    # ✅ Documentation
```

### **1.2 Push to GitHub**

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

---

## **Step 2: Deploy to Streamlit Cloud**

### **2.1 Sign Up / Log In**

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Authorize Streamlit to access your repositories

### **2.2 Create New App**

1. Click **"New app"**
2. Fill in the deployment form:

```
Repository:      your-username/AQI_Predictor
Branch:          main
Main file path:  app/streamlit_app.py
App URL:         aqi-predictor-karachi (or your choice)
```

3. Click **"Advanced settings"** (optional):
   - Python version: **3.11**
   - Custom domain: (if you have one)

4. Click **"Deploy!"**

---

## **Step 3: Configure Secrets**

### **3.1 Open App Settings**

1. Go to your deployed app
2. Click **⋮** (three dots) → **Settings**
3. Navigate to **Secrets** tab

### **3.2 Add Your Secrets**

Paste this configuration (replace with your actual values):

```toml
# AQICN API
AQICN_API_KEY = "your_actual_aqicn_token_here"

# Hopsworks
HOPSWORKS_API_KEY = "your_actual_hopsworks_api_key_here"
HOPSWORKS_PROJECT_NAME = "aqi_predictor99"

# Target City
CITY = "karachi"
CITY_LAT = "24.8607"
CITY_LON = "67.0011"
```

### **3.3 Save & Reboot**

1. Click **"Save"**
2. The app will automatically reboot
3. Wait 2-3 minutes for deployment to complete

---

## **Step 4: Verify Deployment**

### **4.1 Check Dashboard**

Visit your app URL: `https://yourapp.streamlit.app/`

**Expected behavior:**
- ✅ Dashboard loads without errors
- ✅ Current AQI displays
- ✅ 3-day forecast chart shows
- ✅ All navigation pages work

### **4.2 Test Functionality**

1. **Dashboard Page:**
   - Current AQI badge displays
   - Forecast chart shows 4 points (current, 24h, 48h, 72h)
   - Model metrics display

2. **Forecast Details Page:**
   - Table shows all 4 predictions
   - Bar chart displays correctly

3. **Historical Trends:**
   - May show warning if no data (normal - needs backfill)

4. **Feature Importance:**
   - May show warning if SHAP plots not generated yet (normal)

5. **Model Info:**
   - Model metadata displays
   - Feature list shows

### **4.3 Check Logs**

If something fails:
1. Click **⋮** → **Manage app**
2. View logs in the console
3. Look for error messages (especially API key issues)

---

## **Step 5: Common Issues & Solutions**

### **Issue 1: "Failed to fetch predictions"**

**Cause:** Missing or invalid API keys

**Solution:**
1. Go to App Settings → Secrets
2. Verify all keys are correct:
   - `AQICN_API_KEY` (no quotes, just the token)
   - `HOPSWORKS_API_KEY`
   - `HOPSWORKS_PROJECT_NAME`
3. Save and reboot

### **Issue 2: "No trained model found"**

**Cause:** No model in Hopsworks Model Registry

**Solution:**
1. Run training pipeline locally:
   ```bash
   python main.py --pipeline train
   ```
2. Or trigger GitHub Actions training workflow
3. Wait for model to be registered in Hopsworks
4. Dashboard will work once model exists

### **Issue 3: "Module not found" errors**

**Cause:** Missing dependencies or path issues

**Solution:**
1. Check `requirements.txt` includes all packages
2. Verify `sys.path.insert` in `streamlit_app.py`:
   ```python
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
   ```
3. Reboot app

### **Issue 4: "AQICN API rate limit exceeded"**

**Cause:** Too many requests to AQICN API

**Solution:**
1. AQICN free tier: 1000 requests/day
2. Dashboard caches data for 5 minutes (`ttl=300`)
3. If hitting limits, increase cache TTL:
   ```python
   @st.cache_data(ttl=600)  # 10 minutes
   ```

### **Issue 5: "Historical Trends" shows no data**

**Cause:** No historical data in `data/features.csv`

**Solution:**
This is normal on first deployment. Options:
1. Run backfill pipeline locally and upload data
2. Let feature pipeline run for a few days (via GitHub Actions)
3. Dashboard works without historical data

### **Issue 6: Slow loading / timeout**

**Cause:** Hopsworks model download or API calls taking too long

**Solution:**
1. Increase Streamlit timeout in `.streamlit/config.toml`:
   ```toml
   [server]
   maxUploadSize = 200
   maxMessageSize = 200
   enableCORS = false
   ```
2. First load will be slow (downloads model)
3. Subsequent loads use cache (5 min)

---

## **Step 6: Monitoring & Maintenance**

### **6.1 Check App Health**

Regularly visit your dashboard to ensure:
- ✅ Predictions update (cache expires every 5 min)
- ✅ No error messages
- ✅ Model metrics are reasonable

### **6.2 Update App**

When you push changes to GitHub:
1. App auto-deploys on push to `main` branch
2. Takes 2-3 minutes to rebuild
3. Users will see "App is restarting" message

### **6.3 View Analytics**

Streamlit Cloud provides:
- Daily active users
- Page views
- Error rate
- Resource usage

Access via: App Settings → Analytics

### **6.4 Set Up Alerts** (Optional)

Monitor uptime with:
- **UptimeRobot** (free): https://uptimerobot.com/
- **Pingdom** (free tier available)
- **StatusCake** (free plan)

---

## **Step 7: Custom Domain (Optional)**

### **7.1 Add CNAME Record**

In your DNS provider (e.g., Cloudflare, GoDaddy):

```
Type:  CNAME
Name:  aqi (or subdomain of choice)
Value: yourapp.streamlit.app
TTL:   Auto or 3600
```

### **7.2 Configure in Streamlit**

1. Go to App Settings → General
2. Under "Custom domain", enter: `aqi.yourdomain.com`
3. Click "Add domain"
4. Wait 24-48 hours for DNS propagation

### **7.3 Enable HTTPS**

Streamlit Cloud automatically provides SSL certificates via Let's Encrypt.

---

## **Step 8: Advanced Configuration**

### **8.1 Optimize Performance**

**Increase Cache TTL** (if API limits are an issue):
```python
@st.cache_data(ttl=900)  # 15 minutes
def get_inference():
    ...
```

**Reduce Data Loading** (if historical data is large):
```python
# Load only last 30 days by default
history_days = st.sidebar.selectbox("History window", [7, 30], index=1)
```

### **8.2 Add Authentication** (Optional)

Streamlit Cloud Community plans don't support authentication, but you can:

1. **Upgrade to Teams plan** ($250/month) for:
   - Password protection
   - SSO (Google, GitHub, SAML)
   - User management

2. **DIY Authentication:**
   ```python
   import streamlit as st
   
   def check_password():
       if 'password_correct' not in st.session_state:
           st.text_input("Password", type="password", key="password")
           if st.session_state.password == st.secrets["app_password"]:
               st.session_state.password_correct = True
           else:
               st.error("Incorrect password")
               return False
       return True
   
   if not check_password():
       st.stop()
   ```

### **8.3 Enable Dark Mode**

Update `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#ff7e00"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

---

## **Step 9: Cost Considerations**

### **Free Tier Limits:**
- ✅ Unlimited public apps
- ✅ 1 GB resources per app
- ✅ No credit card required
- ❌ No custom authentication
- ❌ Apps sleep after 7 days of inactivity

### **Paid Plans:**

| Feature | Community (Free) | Teams ($250/mo) |
|---------|------------------|-----------------|
| Public apps | Unlimited | Unlimited |
| Private apps | 1 | Unlimited |
| Resources | 1 GB | 4 GB per app |
| Authentication | No | Yes |
| Custom domains | Yes | Yes |
| Support | Community | Priority |

### **API Costs:**
- **AQICN:** FREE (1000 requests/day)
- **OpenMeteo:** FREE (10,000 requests/day, no API key)
- **Hopsworks:** FREE (Feature Store + Model Registry)

**Total cost:** $0/month for entire stack! 🎉

---

## **Step 10: Troubleshooting Checklist**

Before asking for help, verify:

- [ ] All secrets are configured correctly
- [ ] Repository is accessible (public or authorized)
- [ ] `requirements.txt` is complete and up-to-date
- [ ] Main file path is `app/streamlit_app.py`
- [ ] Python version is 3.11
- [ ] No syntax errors in code
- [ ] Model exists in Hopsworks Model Registry
- [ ] API keys are valid and not expired
- [ ] No rate limits hit on AQICN
- [ ] Logs show specific error message

---

## **📚 Additional Resources**

### **Official Documentation:**
- Streamlit Cloud: https://docs.streamlit.io/streamlit-community-cloud
- Secrets Management: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- App Settings: https://docs.streamlit.io/streamlit-community-cloud/manage-your-app

### **Community Support:**
- Streamlit Forum: https://discuss.streamlit.io/
- GitHub Issues: Your repository issues page
- Discord: https://discord.gg/streamlit

### **Related Projects:**
- Hopsworks Examples: https://github.com/logicalclocks/hopsworks-tutorials
- Streamlit Gallery: https://streamlit.io/gallery

---

## **✅ Deployment Success Checklist**

Your deployment is successful when:

- [x] App loads without errors
- [x] Current AQI displays correctly
- [x] 3-day forecast shows **different values** (not just scaled)
- [x] All 5 pages are accessible
- [x] Model info displays correctly
- [x] Footer shows "OpenMeteo" (not "OpenWeatherMap")
- [x] No Python errors in logs
- [x] API keys are working
- [x] Cache is functioning (5-minute TTL)

---

## **🎉 You're Live!**

Your AQI Predictor is now deployed and accessible worldwide!

**Share your dashboard:**
```
https://yourapp.streamlit.app/
```

**Next steps:**
1. Share the link with users
2. Monitor app analytics
3. Iterate based on feedback
4. Keep model updated via GitHub Actions

---

**End of Deployment Guide**

*Happy predicting! 🌍📊*
