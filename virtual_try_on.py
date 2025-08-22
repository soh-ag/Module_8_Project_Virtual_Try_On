import streamlit as st
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

st.set_page_config(layout="wide", page_title="Virtual Try-On")

st.title("Virtual Try-On")

@st.cache_resource
def load_models():
    segmentation_extractor = AutoFeatureExtractor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
    segmentation_model = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
    
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    return segmentation_extractor, segmentation_model, inpaint_pipe

def get_mask(image, region, extractor, model):
    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    mask = None
    if region == 'Upper Region':
        # Corresponds to "Upper-clothes" (4), "Dress" (7)
        mask = torch.isin(pred_seg, torch.tensor([4, 7]))
    else: # Lower Region
        # Corresponds to "Pants" (6), "Skirt" (5)
        mask = torch.isin(pred_seg, torch.tensor([5, 6]))
        
    mask = mask.numpy().astype(np.uint8) * 255
    return Image.fromarray(mask)


segmentation_extractor, segmentation_model, inpaint_pipe = load_models()

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
region = st.sidebar.radio("Select Region", ('Upper Region', 'Lower Region'))
prompt = st.sidebar.text_input("Enter a description of the desired clothing:")
generate_button = st.sidebar.button("Generate")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Input Image")
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True)

with col2:
    st.header("Mask")

with col3:
    st.header("Result")

if generate_button and uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.sidebar.info("Generating mask...")
    mask_image = get_mask(image, region, segmentation_extractor, segmentation_model)
    
    with col2:
        st.image(mask_image, use_column_width=True)
    
    st.sidebar.info("Inpainting...")
    result_image = inpaint_pipe(
        prompt=prompt, 
        image=image.resize((512, 512)), 
        mask_image=mask_image.resize((512, 512))
    ).images[0]
    
    with col3:
        st.image(result_image, use_column_width=True)
    
    st.sidebar.success("Done!")

elif generate_button:
    st.sidebar.error("Please upload an image and enter a prompt.")