import os
import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select
import hugging_face_api
import io
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/cagliostrolab/animagine-xl-3.1"
headers = {"Authorization": HUGGINGFACE_API_KEY}


from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/cagliostrolab/animagine-xl-3.1"
headers = {"Authorization": HUGGINGFACE_API_KEY}


# UI configurations
st.set_page_config(page_title="ReAnime",
                   page_icon=":bridge_at_night:",
                   layout="wide")
# icon.show_icon(":foggy:")
st.markdown("<h1 style='text-align: center; font-size: 100px; color: red;'>REANIME</h1>", unsafe_allow_html=True)


def clearBg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    mask = 255 - mask

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

def printOnTshirt(image):
    tShirts = [cv2.imread('images/white.jpg'),cv2.imread('images/black.png')]
    prints  = []
    for img in tShirts:
        y,x,z=img.shape

        #size of print
        printX = 375
        printY = 400

        #resize print
        image=cv2.resize(image,(printX,printY))

        #Coordinates in tshirt
        yA = int(y/2-(printY/2) + (0.05*y))
        yB = int(y/2+(printY/2) + (0.05*y))
        xA = int(x/2-(printX/2))
        xB = int(x/2+(printX/2))

        #add alpha channel
        alpha1 = np.ones((y,x))*np.float32(255)
        alpha2 = np.ones((printY,printX))*np.float32(255)

        img   = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        img[:,:,3] = alpha1
        image[:,:,3] = alpha2

        mode =1
        if (mode ==0):
            img   = img[:,:,:]*np.float32(1/255)
            image = image[:,:,:]*np.float32(1/255)
            #print(img1[:,:,1])
            img[yA:yB,xA:xB,:] = img[yA:yB,xA:xB,:]*image[:,:,:]
            img = img[:,:,:]*np.float32(255)
        else:
            al =0.1
            img[yA:yB,xA:xB,:] = np.uint8(img[yA:yB,xA:xB,:]*al + image[:,:,:]*(1-al))
        prints.append(img)
    return prints


def configure_searchbar() -> None:
    """
    Setup and display the search elements.

    This function configures the search of the Streamlit application, 
    including the form for user inputs and the resources section.
    """

    with st.form('form'):
        col1, col2,col3 = st.columns([1,5,1])

        with col1:
            pass
        with col3:
            pass
        with col2:
            prompt = st.text_input(
                    "Start typing, Shakespeare ✍🏾",
                    value="Start typing, Shakespeare ✍🏾",
                    placeholder="Start typing, Shakespeare ✍🏾",
                    label_visibility='collapsed'
                )
        col1, col2,col3 = st.columns([3,1,3])
        with col1:
            pass
        with col3:
            pass
        with col2:
            submitted = st.form_submit_button("Generate-Design", type="primary", use_container_width=True)

    st.markdown(
            """
            <style>
            input {
                padding: 0.5rem 1rem !important;
                font-size: 1.5rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    return submitted, prompt

def enhance_prompt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    quality_prompt= "masterpiece, best quality, very aesthetic, absurdres" 
    image_gen_model_format = f"""{quality_prompt}, <number of character><character gender>, <character name>, <series name>, <enhance user query with  keyword (separated by comma) in reference to series>"""
    neg_promt=  "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            # {"role": "system", "content": "You are a ANIME expert, skilled in designing prompt for image generation using the user query convert the query in bew format, if the required information about anime is not given the query use your knowledge."},
            {"role": "user", "content": f"""You are a ANIME expert, skilled in designing prompt for image generation mode. Using the user query enclosed in ## fill the "<>" in the format given in backticks. If the required information about anime is not given the query use your knowledge to complete it.
            
            #{prompt}#
            
            ```{image_gen_model_format}```
            
            """}
    ]
    )
    enhanced_prompt = completion.choices[0].message.content
    enhanced_prompt =enhanced_prompt.replace("```","")
    print(enhanced_prompt)
    return enhanced_prompt, neg_promt
    # pass

gap0 = st.empty()
form_placeholder = st.empty()
gap1 = st.empty()
# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gap2 = st.empty()
gallery_placeholder = st.empty()


def main_page(submitted: bool, prompt: str, negative_prompt: str) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        scheduler (str): Scheduler type for the model.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Scale for classifier-free guidance.
        prompt_strength (float): Prompt strength when using img2img/inpaint.
        refine (str): Refine style to use.
        high_noise_frac (float): Fraction of noise to use for `expert_ensemble_refiner`.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status(':white[👩🏾‍🍳 Whipping up your words into art...]', expanded=True) as status:
            st.write(":white[⚙️ Model initiated]")
            st.write(":white[🙆‍♀️ Stand up and strecth in the meantime]")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        prompt, negative_prompt = enhance_prompt(prompt)
                        
                        def query(payload):
                            response = requests.post(API_URL, headers=headers, json=payload)
                            return response.content
                        image_bytes = query({
                            "inputs": prompt,
                            "negative_prompt":negative_prompt,
                            
                        })
                        if len(image_bytes)<1000:
                            print(image_bytes)
                        
                        image = Image.open(io.BytesIO(image_bytes))
                        if image:
                            st.toast(
                                'Your image has been generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = image

                            # Displaying the image
                            with st.container():
                                st.image(image, caption="Generated Image 🎈")
                        st.session_state.all_images = all_images

                status.update(label="✅ Images generated!",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    with gallery_placeholder.container():
        img = image_select(
            label="",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
            ],
            use_container_width=True
        )

        # col1, col2 = st.columns([1,1])
        # with col1:
        #     st.image("gallery/farmer_sunset.png")

        # with col2:
        #     with st.container():
        #         st.image("gallery/farmer_sunset.png")
            
        #     with st.container():
        #         st.image("gallery/farmer_sunset.png")


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    with gap0.container():
        st.text("")
        st.text("")
    with gap1.container():
        st.text("")
    with gap2.container():
        st.text("")
    
    with form_placeholder.container():
        submitted, prompt = configure_searchbar()
        negative_prompt = ""
        main_page(submitted, prompt, negative_prompt)

if __name__ == "__main__":
    main()
