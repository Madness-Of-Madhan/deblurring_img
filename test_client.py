"""
Test Client for Image Deblurring API
Usage: python test_client.py <blurred_image_path> [reference_image_path]
"""

import requests
import argparse
import json
import base64
from PIL import Image
import io
import os

API_URL = "http://localhost:5000"

def save_base64_image(base64_string, output_path):
    """Decode and save base64 image"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path)
    print(f"Saved: {output_path}")

def test_deblur_api(blur_image_path, reference_image_path=None):
    """Test the /deblur endpoint"""
    
    # Prepare files
    files = {
        'image': open(blur_image_path, 'rb')
    }
    
    if reference_image_path:
        files['reference'] = open(reference_image_path, 'rb')
    
    print(f"\nUploading blurred image: {blur_image_path}")
    if reference_image_path:
        print(f"Reference image: {reference_image_path}")
    
    # Make request
    print("\nSending request to API...")
    response = requests.post(f"{API_URL}/deblur", files=files)
    
    # Close files
    for f in files.values():
        f.close()
    
    if response.status_code == 200:
        print("✓ Request successful!")
        data = response.json()
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        timestamp = data['timestamp']
        
        # Save deblurred images
        print("\nSaving deblurred images...")
        save_base64_image(data['models']['mprnet']['output'], 
                         f'output/mprnet_{timestamp}.png')
        save_base64_image(data['models']['deblurgan_v2']['output'], 
                         f'output/deblurgan_v2_{timestamp}.png')
        save_base64_image(data['models']['unet_patchgan']['output'], 
                         f'output/unet_patchgan_{timestamp}.png')
        
        # Print metrics
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        for model_name, model_key in [('MPRNet', 'mprnet'), 
                                       ('DeblurGAN-v2', 'deblurgan_v2'), 
                                       ('U-Net+PatchGAN', 'unet_patchgan')]:
            metrics = data['models'][model_key]['metrics']
            print(f"\n{model_name}:")
            if metrics:
                if 'psnr' in metrics:
                    print(f"  PSNR:  {metrics['psnr']:.4f} dB")
                if 'ssim' in metrics:
                    print(f"  SSIM:  {metrics['ssim']:.4f}")
                if 'lpips' in metrics:
                    print(f"  LPIPS: {metrics['lpips']:.4f}")
            else:
                print("  No reference image provided - metrics not available")
        
        print("\n" + "="*60)
        
        # Download comparison plots if available
        if data.get('comparison_plot'):
            print("\nDownloading comparison plot...")
            plot_url = f"{API_URL}{data['comparison_plot']}"
            plot_response = requests.get(plot_url)
            if plot_response.status_code == 200:
                with open(f'output/comparison_{timestamp}.png', 'wb') as f:
                    f.write(plot_response.content)
                print(f"Saved: output/comparison_{timestamp}.png")
        
        if data.get('metrics_chart'):
            print("Downloading metrics chart...")
            chart_url = f"{API_URL}{data['metrics_chart']}"
            chart_response = requests.get(chart_url)
            if chart_response.status_code == 200:
                with open(f'output/metrics_{timestamp}.png', 'wb') as f:
                    f.write(chart_response.content)
                print(f"Saved: output/metrics_{timestamp}.png")
        
        print("\n✓ All outputs saved to 'output' directory")
        
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.json())

def test_health():
    """Test the /health endpoint"""
    print("Checking API health...")
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        print("✓ API is healthy")
        print(json.dumps(response.json(), indent=2))
    else:
        print("✗ API health check failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Image Deblurring API')
    parser.add_argument('blur_image', help='Path to blurred image')
    parser.add_argument('--reference', '-r', help='Path to reference/ground truth image (optional)')
    parser.add_argument('--health', action='store_true', help='Check API health')
    
    args = parser.parse_args()
    
    if args.health:
        test_health()
    else:
        if not os.path.exists(args.blur_image):
            print(f"Error: Image file not found: {args.blur_image}")
            exit(1)
        
        if args.reference and not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            exit(1)
        
        test_deblur_api(args.blur_image, args.reference)