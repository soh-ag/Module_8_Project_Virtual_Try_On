
# Virtual Try-On Application

This is a Streamlit web application that allows you to virtually try on clothes. You can upload an image of yourself, select a region (upper or lower body), and provide a text description of a clothing item. The application will then use AI models to generate an image of you wearing the described clothing.

## Installation

1. **Clone the repository or download the code.**

2. **Install the required Python libraries:**

   ```bash
   pip install streamlit torch transformers diffusers Pillow accelerate safetensors
   ```

## How to Run the Application

1. **Open a terminal or command prompt.**

2. **Navigate to the `virtual_try_on` directory:**

   ```bash
   cd path/to/your/virtual_try_on
   ```

3. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

   This will start the web server and open the application in your browser.

## How to Use the Application

1. **Upload your image:** Use the file uploader in the sidebar to upload an image of yourself.

2. **Select Region:** Choose either "Upper Region" or "Lower Region" from the radio buttons in the sidebar.

3. **Enter a Prompt:** In the text box, describe the clothing you want to try on (e.g., "a red t-shirt", "blue jeans").

4. **Generate:** Click the "Generate" button.

5. **View the Results:** The application will display the generated mask and the final image with the new clothing.

## Models Used

*   **Segmentation:** `matei-dorian/segformer-b5-finetuned-human-parsing` from Hugging Face is used to create the segmentation mask of the selected region.
*   **Inpainting:** `runwayml/stable-diffusion-inpainting` from Hugging Face is used to inpaint the new clothing onto the image.
