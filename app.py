# app.py - Gradio Web Application for Road Segmentation
import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from torchvision import transforms
from model import UNet
import config


class RoadSegmentationApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=config.in_channels, out_channels=config.num_classes).to(self.device)
        self.load_model()
        
    def load_model(self):
        """Load the trained model weights"""
        if os.path.exists(config.model_path):
            checkpoint = torch.load(config.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Model loaded successfully")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✓ Model loaded successfully")
            self.model.eval()
        else:
            raise FileNotFoundError(f"No trained model found. Please train the model first.")
    
    def preprocess_image(self, image):
        """Preprocess the input image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((config.img_size, config.img_size))
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)
    
    def predict(self, input_image):
        """Predict road segmentation for the input image"""
        if input_image is None:
            return None, None
        
        img_tensor = self.preprocess_image(input_image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.sigmoid(output).cpu().squeeze().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
        
        pred_image = Image.fromarray(pred_binary, mode='L')
        overlay = self.create_overlay(input_image, pred_binary)
        
        return pred_image, overlay
    
    def create_overlay(self, original_image, mask):
        """Create an overlay of the prediction on the original image"""
        original_resized = original_image.resize((config.img_size, config.img_size))
        original_array = np.array(original_resized).astype(np.float32)
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.float32)
        colored_mask[mask > 127] = [255, 0, 0]
        
        overlay = (original_array * 0.7 + colored_mask * 0.3).astype(np.uint8)
        return Image.fromarray(overlay)


# Initialize the app
try:
    print("\n" + "="*70)
    print("🚀 Initializing Road Segmentation Application")
    print("="*70)
    
    app = RoadSegmentationApp()
    
    print(f"📱 Device: {app.device}")
    print(f"🤖 Model: U-Net")
    print("="*70 + "\n")
    
    # Custom theme with 3-color palette
    # Color 1: #DAA520 (Mustard/Goldenrod)
    # Color 2: #000000 (Black)
    # Color 3: #FFFFFF (White)
    
    theme = gr.themes.Soft(
        primary_hue="yellow",
        secondary_hue="gray",
    )
    
    # Create the Gradio interface
    with gr.Blocks(theme=theme, title="AI Road Segmentation", css="""
        footer {display: none !important;}
        .footer {display: none !important;}
        .svelte-1rjryqp {display: none !important;}
        body {background-color: #FFFFFF !important;}
    """) as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(135deg, #DAA520 0%, #000000 100%);
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white; box-shadow: 0 4px 15px rgba(218, 165, 32, 0.3);">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🛣️ AI Road Segmentation</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                U-Net Deep Learning • Satellite Image Analysis
            </p>
        </div>
        """)
        
        # Main Application
        with gr.Row():
            # Left - Upload Section
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: white; border: 3px solid #000000; border-radius: 12px; 
                            padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(218, 165, 32, 0.2);">
                    <h3 style="margin: 0; color: #000000; font-size: 1.3rem; text-align: center; font-weight: 600;">📤 Upload Satellite Image</h3>
                </div>
                """)
                
                input_image = gr.Image(
                    type="pil",
                    label="",
                    height=400,
                    sources=["upload", "clipboard"],
                    container=False
                )
                
                predict_btn = gr.Button(
                    "🚀 Generate Segmentation",
                    variant="primary",
                    size="lg"
                )
            
            # Right - Results Section
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: white; border: 3px solid #000000; border-radius: 12px; 
                            padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
                    <h3 style="margin: 0; color: #000000; font-size: 1.3rem; text-align: center; font-weight: 600;">🎨 Segmentation Results</h3>
                </div>
                """)
                
                with gr.Tabs():
                    with gr.Tab("Binary Mask"):
                        output_mask = gr.Image(
                            type="pil",
                            label="",
                            height=400
                        )
                    
                    with gr.Tab("Overlay"):
                        output_overlay = gr.Image(
                            type="pil",
                            label="",
                            height=400
                        )
        
        # Connect prediction
        predict_btn.click(
            fn=app.predict,
            inputs=input_image,
            outputs=[output_mask, output_overlay]
        )
    
    if __name__ == "__main__":
        print("🌐 Starting web server...")
        print("🔗 Application: http://127.0.0.1:7860")
        print("⌨️  Press Ctrl+C to stop\n")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease ensure:")
    print("1. Model file 'best_model.pth' exists")
    print("2. All dependencies are installed")
    print("3. Run 'python train.py' if needed\n")
