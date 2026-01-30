# 🚀 Render Free Tier Optimization Guide

## ✅ Changes Made to Fix Memory & Timeout Issues

### 1. Gunicorn Configuration (render.yaml)
```yaml
workers: 1              # Reduced from 2 (saves memory)
threads: 2              # Multi-threading for concurrency
timeout: 300            # Increased to 5 minutes (from 120s)
max-requests: 100       # Restart worker after 100 requests
preload: true           # Load models once, not per worker
```

**Why?**
- Free tier has only 512MB RAM
- Multiple workers = multiple copies of ML models in memory
- 1 worker + 2 threads = better memory usage

### 2. TensorFlow CPU Version
```
tensorflow-cpu>=2.15.0  # Instead of tensorflow
```

**Why?**
- CPU version is 10x smaller than GPU version
- Free tier doesn't have GPU anyway
- Saves ~400MB of memory

### 3. Memory Optimization
- Added `memory_optimizer.py` for garbage collection
- Models loaded once with `--preload` flag
- Automatic cleanup after predictions

---

## 🔍 Current Issues & Solutions

### Issue: Worker Timeout
**Error:** `WORKER TIMEOUT (pid:101)`

**Cause:** 
- ML model prediction taking >120 seconds
- Deep learning model is slow on CPU

**Solutions Applied:**
✅ Increased timeout to 300 seconds
✅ Using tensorflow-cpu (lighter)
✅ Reduced workers to 1

**If still timing out:**
1. Disable deep learning model temporarily
2. Use only Random Forest/XGBoost (faster)

### Issue: Out of Memory
**Error:** `Worker was sent SIGKILL! Perhaps out of memory?`

**Cause:**
- Multiple workers loading all models
- TensorFlow using too much RAM

**Solutions Applied:**
✅ Single worker (1 instead of 2)
✅ TensorFlow-CPU (smaller footprint)
✅ Preload models once

---

## 📊 Memory Usage Breakdown

**Free Tier Limit:** 512 MB RAM

**Estimated Usage:**
- Python + Flask: ~50 MB
- NumPy + Pandas: ~80 MB
- Scikit-learn models: ~100 MB
- TensorFlow-CPU: ~150 MB
- MongoDB driver: ~20 MB
- **Total:** ~400 MB (leaves 112 MB buffer)

---

## 🎯 Performance Optimization Tips

### 1. Disable Heavy Models (If Needed)
Comment out in `requirements.txt`:
```python
# tensorflow-cpu>=2.15.0,<2.18.0
# keras>=2.15.0,<4.0.0
```

App will automatically skip deep learning model.

### 2. Use Lighter Models
- ✅ Logistic Regression (fastest, 1-2s)
- ✅ Random Forest (fast, 2-3s)
- ✅ XGBoost (medium, 3-5s)
- ⚠️ Deep Learning (slow, 10-30s on CPU)

### 3. Lazy Loading (Future Enhancement)
Load models only when requested, not at startup.

---

## 🔧 Troubleshooting

### If App Still Crashes:

**Option 1: Disable TensorFlow**
```bash
# In requirements.txt, comment out:
# tensorflow-cpu>=2.15.0,<2.18.0
# keras>=2.15.0,<4.0.0
# h5py>=3.10.0,<4.0.0
```

**Option 2: Reduce Model Count**
Delete some model files from `data/models/`:
- Keep: `random_forest_*.pkl`, `gradient_boosting_*.pkl`
- Remove: `deep_learning_*.h5`, `lstm_*.h5`

**Option 3: Upgrade Render Plan**
- Starter Plan: $7/month
- 512 MB → 2 GB RAM
- No sleep mode
- Faster CPU

---

## 📈 Monitoring

### Check Render Logs:
```
✓ MongoDB connected successfully
🔄 Loading ML Models...
✅ Total models loaded: 4
```

### Watch for Errors:
```
❌ WORKER TIMEOUT
❌ Out of memory
❌ Model loading failed
```

### Performance Metrics:
- Cold start: 30-60 seconds (first request)
- Warm requests: 2-5 seconds
- Prediction time: 5-30 seconds (depends on model)

---

## 🎯 Recommended Configuration

**For Best Performance on Free Tier:**

1. **Use 2-3 models only:**
   - Random Forest (fast, accurate)
   - XGBoost (best accuracy)
   - Logistic Regression (fastest)

2. **Skip TensorFlow:**
   - Too heavy for free tier
   - Use only if needed

3. **Monitor usage:**
   - Check logs regularly
   - Watch for memory warnings

---

## 🚀 Upgrade Path

**When to upgrade:**
- Consistent timeouts
- Frequent crashes
- Need faster predictions
- More than 100 users/day

**Render Starter ($7/month):**
- 2 GB RAM (4x more)
- No sleep mode
- Faster CPU
- Better for production

---

## ✅ Current Status

After optimization:
- ✅ Workers: 1 (memory efficient)
- ✅ Timeout: 300s (handles slow predictions)
- ✅ TensorFlow: CPU version (lighter)
- ✅ Preload: Models loaded once
- ✅ MongoDB: Connected

**Expected behavior:**
- First request: 30-60s (cold start)
- Subsequent requests: 5-30s (warm)
- No crashes under normal load

---

**Need Help?** Check Render logs or contact support!
