"""
Advanced Test Client for Image Deblurring API
Supports batch processing and detailed reporting
"""

import requests
import argparse
import json
import base64
from PIL import Image
import io
import os
from pathlib import Path
import time
from datetime import datetime

API_URL = "http://localhost:5000"

class DeblurClient:
    def __init__(self, api_url=API_URL):
        self.api_url = api_url
        self.session = requests.Session()
    
    def check_health(self):
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def save_base64_image(self, base64_string, output_path):
        """Decode and save base64 image"""
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img.save(output_path)
        return output_path
    
    def deblur_image(self, blur_path, reference_path=None, output_dir='output'):
        """Deblur a single image"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = {'image': open(blur_path, 'rb')}
        if reference_path:
            files['reference'] = open(reference_path, 'rb')
        
        print(f"\n{'='*70}")
        print(f"Processing: {blur_path}")
        if reference_path:
            print(f"Reference: {reference_path}")
        print('='*70)
        
        start_time = time.time()
        response = self.session.post(f"{self.api_url}/deblur", files=files)
        elapsed_time = time.time() - start_time
        
        for f in files.values():
            f.close()
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
        
        print(f"✅ Request successful! (took {elapsed_time:.2f}s)")
        data = response.json()
        
        timestamp = data['timestamp']
        
        # Save outputs
        print("\n📁 Saving deblurred images...")
        saved_files = []
        
        for model_name, model_key in [('MPRNet', 'mprnet'), 
                                       ('DeblurGAN-v2', 'deblurgan_v2'), 
                                       ('U-Net+PatchGAN', 'unet_patchgan')]:
            output_path = os.path.join(output_dir, f'{model_key}_{timestamp}.png')
            self.save_base64_image(data['models'][model_key]['output'], output_path)
            saved_files.append(output_path)
            print(f"  ✓ {model_name}: {output_path}")
        
        # Print metrics
        if data['has_reference']:
            print("\n📊 Performance Metrics:")
            print(f"{'Model':<20} {'PSNR (dB)':<12} {'SSIM':<12} {'LPIPS':<12}")
            print('-'*56)
            
            for model_name, model_key in [('MPRNet', 'mprnet'), 
                                           ('DeblurGAN-v2', 'deblurgan_v2'), 
                                           ('U-Net+PatchGAN', 'unet_patchgan')]:
                metrics = data['models'][model_key]['metrics']
                psnr = f"{metrics['psnr']:.4f}" if 'psnr' in metrics else 'N/A'
                ssim = f"{metrics['ssim']:.4f}" if 'ssim' in metrics else 'N/A'
                lpips = f"{metrics['lpips']:.4f}" if 'lpips' in metrics else 'N/A'
                print(f"{model_name:<20} {psnr:<12} {ssim:<12} {lpips:<12}")
        
        # Download comparison plots
        if data.get('comparison_plot'):
            print("\n🖼️  Downloading visualizations...")
            plot_path = os.path.join(output_dir, f'comparison_{timestamp}.png')
            self._download_file(data['comparison_plot'], plot_path)
            saved_files.append(plot_path)
            print(f"  ✓ Comparison plot: {plot_path}")
        
        if data.get('metrics_chart'):
            chart_path = os.path.join(output_dir, f'metrics_{timestamp}.png')
            self._download_file(data['metrics_chart'], chart_path)
            saved_files.append(chart_path)
            print(f"  ✓ Metrics chart: {chart_path}")
        
        return {
            'data': data,
            'saved_files': saved_files,
            'elapsed_time': elapsed_time
        }
    
    def _download_file(self, url_path, output_path):
        """Download file from server"""
        url = f"{self.api_url}{url_path}"
        response = self.session.get(url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
    
    def batch_deblur(self, blur_dir, reference_dir=None, output_dir='batch_output'):
        """Process multiple images in batch"""
        blur_path = Path(blur_dir)
        reference_path = Path(reference_dir) if reference_dir else None
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        blur_images = [f for f in blur_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not blur_images:
            print(f"No images found in {blur_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING - {len(blur_images)} images")
        print('='*70)
        
        results = []
        failed = []
        
        for idx, blur_img in enumerate(blur_images, 1):
            print(f"\n[{idx}/{len(blur_images)}] Processing {blur_img.name}...")
            
            ref_img = None
            if reference_path:
                ref_img = reference_path / blur_img.name
                if not ref_img.exists():
                    print(f"⚠️  Warning: No matching reference image found")
                    ref_img = None
            
            try:
                result = self.deblur_image(
                    str(blur_img), 
                    str(ref_img) if ref_img else None,
                    output_dir
                )
                if result:
                    results.append(result)
                else:
                    failed.append(blur_img.name)
            except Exception as e:
                print(f"❌ Error processing {blur_img.name}: {e}")
                failed.append(blur_img.name)
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print('='*70)
        print(f"✅ Successful: {len(results)}/{len(blur_images)}")
        print(f"❌ Failed: {len(failed)}/{len(blur_images)}")
        
        if failed:
            print("\nFailed images:")
            for img in failed:
                print(f"  - {img}")
        
        if results and results[0]['data']['has_reference']:
            print("\n📊 Average Metrics:")
            self._print_average_metrics(results)
        
        # Save summary report
        self._save_batch_report(results, failed, output_dir)
    
    def _print_average_metrics(self, results):
        """Calculate and print average metrics"""
        model_keys = ['mprnet', 'deblurgan_v2', 'unet_patchgan']
        model_names = ['MPRNet', 'DeblurGAN-v2', 'U-Net+PatchGAN']
        
        print(f"{'Model':<20} {'Avg PSNR':<12} {'Avg SSIM':<12} {'Avg LPIPS':<12}")
        print('-'*56)
        
        for model_name, model_key in zip(model_names, model_keys):
            psnr_values = []
            ssim_values = []
            lpips_values = []
            
            for result in results:
                metrics = result['data']['models'][model_key]['metrics']
                if 'psnr' in metrics:
                    psnr_values.append(metrics['psnr'])
                if 'ssim' in metrics:
                    ssim_values.append(metrics['ssim'])
                if 'lpips' in metrics:
                    lpips_values.append(metrics['lpips'])
            
            avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
            avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
            avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else 0
            
            print(f"{model_name:<20} {avg_psnr:<12.4f} {avg_ssim:<12.4f} {avg_lpips:<12.4f}")
    
    def _save_batch_report(self, results, failed, output_dir):
        """Save batch processing report"""
        report_path = os.path.join(output_dir, f'batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(results),
            'total_failed': len(failed),
            'failed_images': failed,
            'results': []
        }
        
        for result in results:
            report['results'].append({
                'timestamp': result['data']['timestamp'],
                'elapsed_time': result['elapsed_time'],
                'has_reference': result['data']['has_reference'],
                'metrics': {
                    'mprnet': result['data']['models']['mprnet']['metrics'],
                    'deblurgan_v2': result['data']['models']['deblurgan_v2']['metrics'],
                    'unet_patchgan': result['data']['models']['unet_patchgan']['metrics']
                }
            })
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Image Deblurring API Client')
    parser.add_argument('blur_image', nargs='?', help='Path to blurred image or directory for batch processing')
    parser.add_argument('--reference', '-r', help='Path to reference image or directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch processing mode')
    parser.add_argument('--health', action='store_true', help='Check API health')
    parser.add_argument('--url', default=API_URL, help='API URL')
    
    args = parser.parse_args()
    
    client = DeblurClient(args.url)
    
    if args.health:
        print("Checking API health...")
        if client.check_health():
            print("✅ API is healthy and ready!")
            response = client.session.get(f"{args.url}/health")
            print(json.dumps(response.json(), indent=2))
        else:
            print("❌ API is not responding. Make sure the server is running.")
        return
    
    if not args.blur_image:
        parser.print_help()
        return
    
    # Check API health first
    if not client.check_health():
        print("❌ Error: API is not responding. Make sure the server is running.")
        print(f"   Expected URL: {args.url}")
        return
    
    if args.batch:
        if not os.path.isdir(args.blur_image):
            print(f"Error: {args.blur_image} is not a directory")
            return
        client.batch_deblur(args.blur_image, args.reference, args.output)
    else:
        if not os.path.exists(args.blur_image):
            print(f"Error: Image file not found: {args.blur_image}")
            return
        
        if args.reference and not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            return
        
        client.deblur_image(args.blur_image, args.reference, args.output)

if __name__ == "__main__":
    main()