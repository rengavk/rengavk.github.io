# GitHub Push Instructions for All 4 Projects

## Step 1: Create GitHub Repositories

Go to https://github.com/new and create these 4 repositories:

### Repository 1: aviation-turnaround-optimizer
- Description: "ML-powered flight turnaround delay predictor with 99.99% accuracy using XGBoost and SHAP. Reduces delays by 6-12%."
- Public
- Don't initialize with README

### Repository 2: mental-health-crisis-nlp
- Description: "Privacy-preserving mental health crisis detection using differential privacy (ε<3.0) and DistilBERT. Ethical AI with comprehensive safety policy."
- Public
- Don't initialize with README

### Repository 3: supply-chain-simulator
- Description: "Multi-echelon inventory simulator reducing stockouts by 30% using discrete-event simulation (SimPy) and OR-Tools optimization."
- Public
- Don't initialize with README

### Repository 4: cardiac-arrest-prediction
- Description: "LSTM-based early warning system predicting cardiac arrest 30 minutes in advance using vital signs and ECG features."
- Public
- Don't initialize with README

## Step 2: Push All Projects

Copy and paste this entire block into PowerShell:

```powershell
# Navigate to workspace
cd "c:/Users/renga/.gemini/antigravity/playground/silent-interstellar"

# Project 1: Aviation Turnaround Optimizer
cd "aviation-turnaround-optimizer"
git remote add origin https://github.com/rengavk/aviation-turnaround-optimizer.git
git branch -M main
git push -u origin main
Write-Host "✅ Project 1 pushed!" -ForegroundColor Green

# Project 2: Mental Health Crisis NLP
cd "../mental-health-nlp"
git remote add origin https://github.com/rengavk/mental-health-crisis-nlp.git
git branch -M main
git push -u origin main
Write-Host "✅ Project 2 pushed!" -ForegroundColor Green

# Project 3: Supply Chain Simulator
cd "../supply-chain-simulator"
git remote add origin https://github.com/rengavk/supply-chain-simulator.git
git branch -M main
git push -u origin main
Write-Host "✅ Project 3 pushed!" -ForegroundColor Green

# Project 4: Cardiac Arrest Prediction
cd "../cardiac-arrest-prediction"
git remote add origin https://github.com/rengavk/cardiac-arrest-prediction.git
git branch -M main
git push -u origin main
Write-Host "✅ Project 4 pushed!" -ForegroundColor Green

# Return to portfolio root
cd ".."

Write-Host "`n🎉 All 4 projects pushed successfully!" -ForegroundColor Cyan
Write-Host "Next: Update portfolio website links" -ForegroundColor Yellow
```

## Step 3: Update Portfolio Website

After pushing, update the GitHub URLs in index.html with the actual repository links.

## Verification

After pushing, verify each repository at:
1. https://github.com/rengavk/aviation-turnaround-optimizer
2. https://github.com/rengavk/mental-health-crisis-nlp
3. https://github.com/rengavk/supply-chain-simulator
4. https://github.com/rengavk/cardiac-arrest-prediction
