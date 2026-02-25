# API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Request/Response Format](#requestresponse-format)
5. [Error Handling](#error-handling)
6. [Code Examples](#code-examples)
7. [Rate Limiting](#rate-limiting)

## Overview

The Image Deblurring API provides endpoints for processing blurred images using three different deep learning models:
- MPRNet (Multi-Stage Progressive Restoration Network)
- DeblurGAN-v2
- U-Net + PatchGAN

**Base URL**: `http://localhost:5000`

**API Version**: 1.0

## Authentication

Currently, the API does not require authentication. This may change in future versions.

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

**Endpoint**: `GET /`

**Response**:
```json
{
  "name": "Image Deblurring API",
  "version": "1.0",
  "models": ["MPRNet", "DeblurGAN-v2", "U-Net+PatchGAN"],
  "endpoints": {
    "/deblur": "POST - Upload blurred image for deblurring",
    "/health": "GET - Check API health status"
  }
}
```

### 2. Health Check

Check if the API is running and models are loaded.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": true
}
```

### 3. Deblur Image

Process a blurred image with all three models.

**Endpoint**: `POST /deblur`

**Content-Type**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | Blurred input image (JPEG, PNG, etc.) |
| reference | File | No | Reference/ground truth image for metrics calculation |

**Example Request (cURL)**:
```bash
curl -X POST \
  -F "image=@blurred_image.jpg" \
  -F "reference=@sharp_image.jpg" \
  http://localhost:5000/deblur
```

**Response**: See [Response Format](#response-format) section below.

### 4. Static Files

Retrieve generated comparison plots and metrics charts.

**Endpoint**: `GET /static/<path:filename>`

**Example**:
```
GET /static/comparisons/comparison_20240129_143052.png
```

## Request/Response Format

### Response Format

#### Success Response (200 OK)

```json
{
  "timestamp": "20240129_143052_123456",
  "input_image": "base64_encoded_string...",
  "models": {
    "mprnet": {
      "output": "base64_encoded_string...",
      "metrics": {
        "psnr": 25.4321,
        "ssim": 0.8765,
        "lpips": 0.1234
      }
    },
    "deblurgan_v2": {
      "output": "base64_encoded_string...",
      "metrics": {
        "psnr": 26.1234,
        "ssim": 0.8912,
        "lpips": 0.1123
      }
    },
    "unet_patchgan": {
      "output": "base64_encoded_string...",
      "metrics": {
        "psnr": 24.9876,
        "ssim": 0.8654,
        "lpips": 0.1345
      }
    }
  },
  "comparison_plot": "/static/comparisons/comparison_20240129_143052_123456.png",
  "metrics_chart": "/static/comparisons/metrics_20240129_143052_123456.png",
  "has_reference": true
}
```

**Response Fields**:

- `timestamp`: Unique identifier for this processing session
- `input_image`: Base64-encoded blurred input image
- `models`: Object containing results from each model
  - `output`: Base64-encoded deblurred image
  - `metrics`: Performance metrics (only if reference provided)
    - `psnr`: Peak Signal-to-Noise Ratio (higher is better)
    - `ssim`: Structural Similarity Index (0-1, closer to 1 is better)
    - `lpips`: Learned Perceptual Image Patch Similarity (lower is better)
- `comparison_plot`: URL to visual comparison of all models
- `metrics_chart`: URL to metrics bar chart (only if reference provided)
- `has_reference`: Boolean indicating if metrics were calculated

#### Error Response (4xx, 5xx)

```json
{
  "error": "Error message describing what went wrong"
}
```

## Error Handling

### Common Error Codes

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 400 | Bad Request | No image file provided, empty filename |
| 500 | Internal Server Error | Model inference error, file processing error |

### Error Response Examples

**No Image Provided**:
```json
{
  "error": "No image file provided"
}
```

**Processing Error**:
```json
{
  "error": "CUDA out of memory"
}
```

## Code Examples

### Python (requests)

```python
import requests
import base64
from PIL import Image
import io

# Simple deblurring
with open('blurred.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/deblur', files=files)

data = response.json()

# Decode and save results
for model_name in ['mprnet', 'deblurgan_v2', 'unet_patchgan']:
    img_data = base64.b64decode(data['models'][model_name]['output'])
    img = Image.open(io.BytesIO(img_data))
    img.save(f'{model_name}_output.png')
```

### Python (with reference image)

```python
import requests

files = {
    'image': open('blurred.jpg', 'rb'),
    'reference': open('sharp.jpg', 'rb')
}

response = requests.post('http://localhost:5000/deblur', files=files)
data = response.json()

# Access metrics
print(f"MPRNet PSNR: {data['models']['mprnet']['metrics']['psnr']:.4f}")
print(f"DeblurGAN SSIM: {data['models']['deblurgan_v2']['metrics']['ssim']:.4f}")
```

### JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('http://localhost:5000/deblur', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Deblurring complete!');
    
    // Display MPRNet output
    const img = new Image();
    img.src = 'data:image/png;base64,' + data.models.mprnet.output;
    document.body.appendChild(img);
})
.catch(error => console.error('Error:', error));
```

### cURL

**Basic usage**:
```bash
curl -X POST \
  -F "image=@blurred.jpg" \
  http://localhost:5000/deblur \
  | jq .
```

**With reference image**:
```bash
curl -X POST \
  -F "image=@blurred.jpg" \
  -F "reference=@sharp.jpg" \
  http://localhost:5000/deblur \
  | jq '.models.mprnet.metrics'
```

**Save response to file**:
```bash
curl -X POST \
  -F "image=@blurred.jpg" \
  http://localhost:5000/deblur \
  -o response.json
```

## Rate Limiting

Currently, there is no rate limiting implemented. Future versions may include:
- Request rate limits per IP
- Concurrent processing limits
- Queue management for batch processing

## Performance Considerations

### Image Size
- Default processing size: 256x256 pixels
- Larger images are resized, which may affect quality
- Processing time increases with image size

### GPU vs CPU
- GPU processing is significantly faster
- CPU fallback is available but slower
- Batch processing benefits greatly from GPU

### Concurrent Requests
- Multiple concurrent requests are supported
- Each request uses GPU/CPU resources
- Consider system resources when sending batch requests

## Best Practices

1. **Always check health endpoint** before processing
2. **Include reference images** when possible for metrics
3. **Handle base64 properly** when decoding images
4. **Save comparison plots** for visual analysis
5. **Use batch processing** for multiple images

## Metrics Interpretation

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in decibels (dB)
- Higher values indicate better quality
- Typical range: 20-40 dB
- > 30 dB is generally considered good

### SSIM (Structural Similarity Index)
- Range: 0 to 1
- Closer to 1 indicates better structural similarity
- > 0.9 is considered excellent
- More perceptually relevant than PSNR

### LPIPS (Learned Perceptual Image Patch Similarity)
- Lower values indicate better perceptual quality
- Range: typically 0 to 1
- Based on deep features
- Better correlated with human perception

## Troubleshooting

### Issue: Slow processing
**Solution**: 
- Ensure GPU is available and being used
- Reduce image size
- Process images in smaller batches

### Issue: Out of memory errors
**Solution**:
- Reduce image resolution
- Use CPU mode if GPU memory is limited
- Process one image at a time

### Issue: Poor deblurring quality
**Solution**:
- Models need proper training weights
- Current implementation uses random initialization
- Load pre-trained weights for production use

## Support

For additional support:
- Check the README.md file
- Review error messages carefully
- Ensure all dependencies are installed
- Verify API server is running

## Future Enhancements

Planned features for future versions:
- Model selection (choose specific models)
- Custom model weight loading
- Batch processing endpoint
- WebSocket support for real-time updates
- Authentication and API keys
- Rate limiting
- Additional metrics and quality assessments