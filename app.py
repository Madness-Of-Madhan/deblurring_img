from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import logging
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # required for Flask (no GUI)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Import models
from models.mprnet import MPRNet 
from models.deblurgan_v2_gopro import DeblurGANv2Generator
from models.unet_patchgan_deblurgan_colab import UNetGenerator
from inference.engine import InferenceEngine
from inference.preprocess import preprocess
from inference.postprocess import tensor_to_base64

# ===================== SETUP =====================
app = Flask(__name__)
CORS(app)  # 🔥 CRITICAL: Enable CORS for frontend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create static directory for future comparison plots
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
logger.info(f"Using device: {device}")

CLASSIFIER_MODEL_PATH = "weights/fine_tuned_best.h5"

try:
    logger.info("Loading Blur Type Classifier...")
    classifier_model = load_model(CLASSIFIER_MODEL_PATH)
    logger.info("✅ Blur classifier loaded")
except Exception as e:
    logger.error(f"❌ Failed to load classifier: {e}")
    classifier_model = None

CLASS_NAMES = ["Sharp", "Gaussian Blur", "Motion Blur"]

# ===================== LOAD MODELS =====================
try:
    logger.info("Loading MPRNet...")
    mprnet = MPRNet().to(device)
    mprnet.load_state_dict(
        torch.load("weights/mprnet_gopro.pth", map_location=device, weights_only=True)
    )
    mprnet.eval()
    logger.info("✅ MPRNet loaded")

    logger.info("Loading DeblurGAN-v2...")
    deblurgan = DeblurGANv2Generator().to(device)
    deblurgan.load_state_dict(
        torch.load("weights/deblurgan_v2_gopro.pth", map_location=device, weights_only=True)
    )
    deblurgan.eval()
    logger.info("✅ DeblurGAN-v2 loaded")

    logger.info("Loading U-Net...")
    unet = UNetGenerator().to(device)
    unet.load_state_dict(
        torch.load("weights/unet_patchgan_gopro.pth", map_location=device, weights_only=True)
    )
    unet.eval()
    logger.info("✅ U-Net loaded")

    engine = InferenceEngine(mprnet, deblurgan, unet)
    logger.info("✅ Inference engine ready")

except Exception as e:
    logger.error(f"❌ Failed to load models: {e}")
    sys.exit(1)

# ===================== WARMUP =====================
try:
    logger.info("Warming up models...")
    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=device)
        engine.run_all(dummy)
    logger.info("✅ Warmup complete")
except Exception as e:
    logger.warning(f"⚠️ Warmup failed: {e}")

# ===================== HEALTH CHECK =====================
@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "models_loaded": True
    }), 200

# ===================== SERVE STATIC FILES =====================
@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve comparison plots and charts (for future use)"""
    return send_from_directory(STATIC_DIR, filename)

# ===================== DEBLUR API =====================
@app.route("/deblur", methods=["POST"])
def deblur():
    """
    Main deblurring endpoint
    
    Expected frontend response format:
    {
        "input_image": "base64_string",
        "models": {
            "mprnet": {"output": "base64", "metrics": {}},
            "deblurgan_v2": {"output": "base64", "metrics": {}},
            "unet_patchgan": {"output": "base64", "metrics": {}}
        },
        "comparison_plot": "/static/comparison.png",  # Optional
        "metrics_chart": "/static/metrics.png"        # Optional
    }
    """
    
    # Validate request
    if "image" not in request.files:
        logger.warning("Request missing image file")
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files["image"]
    # Run blur classification first
    classification_result = None
    if classifier_model:
        classification_result = classify_blur(image_file)

    reference_file = request.files.get("reference")  # Optional reference image
    
    # Validate file type
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    if not any(image_file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, or WEBP"}), 400

    try:
        logger.info(f"Processing image: {image_file.filename}")
        
        # Preprocess input image
        x = preprocess(image_file, device)
        logger.info(f"Input shape: {x.shape}")
        
        # Get input image as base64 for display
        input_base64 = tensor_to_base64(x)

        # Run inference on all models
        with torch.no_grad():
            outputs = engine.run_all(x)
        
        logger.info("✅ Inference complete")

        # Build response matching frontend expectations
        response = {
            "input_image": input_base64,
            "classification": classification_result,
            "models": {
                "mprnet": {
                    "output": tensor_to_base64(outputs["mprnet"]),
                    "metrics": {}
                },
                "deblurgan_v2": {
                    "output": tensor_to_base64(outputs["deblurgan_v2"]),
                    "metrics": {}
                },
                "unet_patchgan": {
                    "output": tensor_to_base64(outputs["unet_patchgan"]),
                    "metrics": {}
                }
            }
        }
        
        # Calculate metrics if reference image provided
        if reference_file:
            try:
                logger.info("Processing reference image for metrics...")
                reference = preprocess(reference_file, device)
                
                # Calculate metrics for each model
                all_metrics = {}

                for model_name, output_tensor in outputs.items():
                    metrics = calculate_metrics(output_tensor, reference)
                    response["models"][model_name]["metrics"] = metrics
                    all_metrics[model_name] = metrics
                
                logger.info("✅ Metrics calculated")
                graph_path = STATIC_DIR / "metrics_comparison.png"
                generate_metrics_graph(all_metrics, graph_path)
                response["metrics_chart"] = "/static/metrics_comparison.png"

                
            except Exception as e:
                logger.warning(f"⚠️ Failed to calculate metrics: {e}")
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# ===================== METRICS CALCULATION =====================
def calculate_metrics(output, reference):
    """
    Calculate PSNR, SSIM, LPIPS metrics
    
    Install required packages:
    pip install scikit-image lpips
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        import numpy as np
        
        # Convert tensors to numpy arrays [H, W, C]
        output_np = output.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
        reference_np = reference.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)

        
        # Ensure values are in [0, 1] range
        output_np = (output_np + 1) / 2
        reference_np = (reference_np + 1) / 2
        output_np = np.clip(output_np, 0, 1)
        reference_np = np.clip(reference_np, 0, 1)
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(reference_np, output_np, data_range=1.0)
        
        # Calculate SSIM
        ssim = structural_similarity(
        reference_np, 
            output_np, 
            data_range=1.0, 
            channel_axis=2
        )
        
        # LPIPS calculation (optional - requires more setup)
        # Uncomment if you have lpips installed:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        lpips_value = loss_fn(output, reference).item()
        lpips_value = 0.0  # Placeholder
        
        return {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips_value)
        }
        
    except ImportError:
        logger.warning("⚠️ scikit-image not installed. Install with: pip install scikit-image")
        return {}
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {}
    
def generate_metrics_graph(all_metrics, save_path):
        models = list(all_metrics.keys())

        psnr = [all_metrics[m]["psnr"] for m in models]
        ssim = [all_metrics[m]["ssim"] for m in models]
        lpips = [all_metrics[m]["lpips"] for m in models]

        x = range(len(models))

        plt.figure(figsize=(8, 4))
        plt.bar(x, psnr, width=0.25, label="PSNR")
        plt.bar([i + 0.25 for i in x], ssim, width=0.25, label="SSIM")
        plt.bar([i + 0.5 for i in x], lpips, width=0.25, label="LPIPS")

        plt.xticks([i + 0.25 for i in x], models)
        plt.ylabel("Metric Value")
        plt.title("Model Comparison Metrics")
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()

# ===================== BLUR TYPE PREDICTION =====================
def classify_blur(image_file):
    try:
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = classifier_model.predict(image_array, verbose=0)
        pred_class = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_class])

        return {
            "label": CLASS_NAMES[pred_class],
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return None


# ===================== ERROR HANDLERS =====================
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size: 16MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ===================== RUN =====================
if __name__ == "__main__":
    logger.info("🚀 Starting Flask server on http://localhost:5000")
    logger.info("📱 Open your frontend HTML file in a browser to test")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # Set to True only for local development
        threaded=True
    )