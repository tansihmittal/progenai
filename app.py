import streamlit as st
import numpy as np
from PIL import Image
import rembg
from diffusers import StableDiffusionInpaintPipeline
import torch
import io
import base64

class AdvancedProductPhotoGenerator:
    def __init__(self):
        self.rembg_session = rembg.new_session()
        self.model = self._load_model()

    def _load_model(self):
        """Load and cache model"""
        model = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting"
        )
        return model

    def preprocess_image(self, image):
        """Advanced image preprocessing"""
        # Resize to optimal dimensions
        image = image.resize((512, 512), Image.LANCZOS)
        return image

    def remove_background(self, image):
        """Improved background removal"""
        processed_image = self.preprocess_image(image)
        return rembg.remove(processed_image, session=self.rembg_session)

    def prepare_image_and_mask(self, transparent_image):
        """Robust image and mask preparation"""
        image_rgb = transparent_image.convert("RGB")
        np_image = np.array(transparent_image)
        
        mask = np.zeros(image_rgb.size[::-1], dtype=np.uint8)
        if np_image.shape[2] == 4:
            mask = (np_image[:, :, 3] == 0).astype(np.uint8) * 255
        
        return image_rgb, Image.fromarray(mask)

    def generate_background(self, transparent_image, prompt, style='studio', seed=None):
        """Enhanced background generation with style options"""
        # Predefined style prompts
        style_prompts = {
            'studio': "Clean white studio background, soft professional lighting",
            'luxury': "Elegant marble surface, soft gradient lighting",
            'minimal': "Minimalist white background, soft shadows",
            'natural': "Soft natural light, subtle outdoor texture"
        }

        # Combine user prompt with style
        full_prompt = f"{style_prompts.get(style, style_prompts['studio'])}, {prompt}"
        
        # Set random seed for reproducibility
        generator = torch.Generator().manual_seed(
            seed if seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
        )
        
        image_rgb, mask_image = self.prepare_image_and_mask(transparent_image)
        
        generated_image = self.model(
            prompt=full_prompt,
            image=image_rgb,
            mask_image=mask_image,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        return generated_image

    def generate_variations(self, image, prompt, num_variations=3):
        """Generate multiple image variations"""
        variations = []
        for _ in range(num_variations):
            # Use torch for seed generation
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            variation = self.generate_background(image, prompt, seed=seed)
            variations.append(variation)
        return variations

    def run(self):
        st.title("🖼️ Advanced Product Photo Generator")
        
        uploaded_file = st.file_uploader("Upload Product Image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image")
            
            # Style and prompt selection
            col1, col2 = st.columns(2)
            with col1:
                style = st.selectbox("Background Style", 
                    ['Studio', 'Luxury', 'Minimal', 'Natural']
                ).lower()
            
            with col2:
                num_variations = st.slider("Number of Variations", 1, 4, 3)
            
            prompt = st.text_input("Custom Prompt", 
                placeholder="Additional description for background")
            
            if st.button("Generate Product Photos"):
                with st.spinner("Processing images..."):
                    # Remove Background
                    transparent_image = self.remove_background(original_image)
                    st.image(transparent_image, caption="Background Removed")
                    
                    # Generate Variations
                    generated_images = self.generate_variations(
                        transparent_image, 
                        prompt, 
                        num_variations
                    )
                    
                    # Display Generated Images
                    cols = st.columns(num_variations)
                    for i, img in enumerate(generated_images):
                        with cols[i]:
                            st.image(img, caption=f"Variation {i+1}")

def main():
    generator = AdvancedProductPhotoGenerator()
    generator.run()

if __name__ == "__main__":
    main()