# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the Server

```bash
python app.py
```

You should see:
```
Image Deblurring API Server
==================================================
Device: cuda
Models loaded: MPRNet, DeblurGAN-v2, U-Net+PatchGAN
==================================================

 * Running on http://0.0.0.0:5000
```

### Step 3: Test the API

**Option A: Use the Web Interface**

1. Open your browser
2. Navigate to `http://localhost:5000`
3. Upload a blurred image
4. Click "Deblur Image"
5. View results!

**Option B: Use the Command Line**

```bash
# Basic usage
python test_client.py path/to/blurred_image.jpg

# With reference image for metrics
python test_client.py blurred.jpg --reference sharp.jpg

# Batch processing
python advanced_client.py blur_folder/ --batch --reference sharp_folder/
```

**Option C: Use cURL**

```bash
curl -X POST -F "image=@blurred.jpg" http://localhost:5000/deblur > result.json
```

## 📂 File Structure

```
deblurring-api/
├── app.py                      # Main Flask application ⭐
├── requirements.txt            # Python dependencies
├── test_client.py             # Simple test client
├── advanced_client.py         # Advanced batch client
├── templates/
│   └── index.html             # Web interface
├── static/
│   ├── results/               # Deblurred images
│   └── comparisons/           # Comparison plots
├── output/                    # Client output (auto-created)
├── README.md                  # Full documentation
├── API_DOCUMENTATION.md       # API reference
├── QUICK_START.md             # This file
├── Dockerfile                 # Docker container
└── docker-compose.yml         # Docker compose
```

## 🎯 Common Use Cases

### Use Case 1: Single Image Deblurring

```bash
python test_client.py my_blurry_photo.jpg
```

Output:
- `output/mprnet_*.png`
- `output/deblurgan_v2_*.png`
- `output/unet_patchgan_*.png`
- `output/comparison_*.png`

### Use Case 2: Quality Evaluation

```bash
python test_client.py blurred.jpg --reference ground_truth.jpg
```

You'll get:
- Deblurred images
- PSNR, SSIM, LPIPS metrics
- Metrics comparison chart

### Use Case 3: Batch Processing

```bash
python advanced_client.py ./blur_images/ --batch --output batch_results/
```

Processes all images in the folder and generates a summary report.

### Use Case 4: Web Interface

1. Start server: `python app.py`
2. Open browser: `http://localhost:5000`
3. Drag & drop your images
4. Compare results visually

## 🐳 Docker Quick Start

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:5000
```

## ⚡ Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster processing
2. **Batch Processing**: Use advanced_client.py for multiple images
3. **Image Size**: Smaller images process faster (default: 256x256)

## 🔧 Quick Troubleshooting

**Problem**: "Connection refused"
- **Solution**: Make sure `python app.py` is running

**Problem**: "CUDA out of memory"
- **Solution**: Reduce batch size or use CPU mode

**Problem**: "Port 5000 already in use"
- **Solution**: Edit `app.py` and change port to 8000

## 📊 Understanding the Results

### Metrics Explained (when reference image provided)

- **PSNR**: Higher is better (>30 dB is good)
- **SSIM**: Closer to 1.0 is better (>0.9 is excellent)
- **LPIPS**: Lower is better (<0.2 is good)

### Visual Comparison

The comparison plot shows:
- Top-left: Your blurred input
- Top-right: MPRNet output
- Bottom-left: DeblurGAN-v2 output
- Bottom-right: U-Net+PatchGAN output

## 🎨 Example Workflow

```bash
# 1. Check API health
python test_client.py --health

# 2. Test with one image
python test_client.py sample_blur.jpg

# 3. Evaluate with reference
python test_client.py sample_blur.jpg -r sample_sharp.jpg

# 4. Process a batch
python advanced_client.py ./test_images/ --batch

# 5. View results
open output/comparison_*.png
```

## 📚 Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for API reference
- Train models with your own data for better results
- Load pre-trained weights for production use

## 💡 Tips

1. Start with the web interface - it's the easiest
2. Use the test_client.py for scripting
3. Use advanced_client.py for batch jobs
4. Always provide reference images when available for metrics

## 🆘 Need Help?

- Check error messages carefully
- Ensure all dependencies are installed
- Make sure the server is running
- Try the health check: `python test_client.py --health`

---

**Ready to deblur? Start with:** `python app.py` 🚀