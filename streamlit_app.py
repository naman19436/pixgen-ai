import os
import replicate
import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select
# from streamlit_carousel import carousel
import hugging_face_api
import io
import cv2
import base64
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
# from screeninfo import get_monitors
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
st.set_page_config(page_title="AniKai",
                   page_icon=":bridge_at_night:",
                   layout="wide")
# icon.show_icon(":foggy:")
st.markdown("<h1 style='text-align: center; font-size: 100px; color: red;'>AniKai</h1></h1>", unsafe_allow_html=True)


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
    tShirts = [cv2.cvtColor(cv2.imread('images/white.jpg'),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread('images/black.png'),cv2.COLOR_BGR2RGB)]
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
        prints.append(Image.fromarray(img))
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
                    value="",
                    placeholder="Re-imagine your favorite Anime",
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
    quality_prompt= "masterpiece" #, best quality, very aesthetic, absurdres" 
    image_gen_model_format = f"""{quality_prompt}, <number of character><character gender>, <character name>, <series name>, <rest of the user query>"""
    neg_promt=  "artistic error, nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits,  username, scan, [abstract]"
    
    messages = [
            # {"role": "system", "content": "You are a ANIME expert, skilled in designing prompt for image generation using the user query convert the query in bew format, if the required information about anime is not given the query use your knowledge."},
            {"role": "user", "content": f"""You are a ANIME expert, skilled in designing prompt for image generation mode. Using the user query enclosed in ## fill the "<>" in the format given in backticks. If the required information about anime is not given the query use your knowledge to complete it.
            
            #{prompt}#
            
            ```{image_gen_model_format}```
            
            You MUST give numeric value(like 1,2,3) for <character number> and ONLY choose <character gender> from boy or girl
            """

            }
        ]
      
    messages = [
            # {"role": "system", "content": "You are a ANIME expert, skilled in designing prompt for image generation using the user query convert the query in bew format, if the required information about anime is not given the query use your knowledge."},
            {"role": "user", "content": f""" description for gojou satoru is "1boy, male focus, gojou satoru, jujutsu kaisen, black jacket, blindfold lift, blue eyes, glowing, glowing eyes, high collar, jacket, jujutsu tech uniform, solo, grin, white hair"
give similar for {prompt}. ONLY give description and enclosed it in three backticks."""
            }
        ]
    
    completion = client.chat.completions.create(
        # model="gpt-3.5-turbo", #"gpt-4o",
        model="gpt-4o",
        messages=messages
    )
    enhanced_prompt = completion.choices[0].message.content
    enhanced_prompt =enhanced_prompt.replace("`","")
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
        with st.status(':gray[👩🏾‍🍳 Whipping up your words into art...]', expanded=False) as status:
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
                        image_art = Image.open(io.BytesIO(image_bytes))
                        image = np.array(image_art)
                        image_white, image_black = printOnTshirt(image)
                        image_white.save("gallery/white.png")
                        image_black.save("gallery/black.png")
                        all_images  = [image_art, image_white, image_black]
                        image = all_images[0]
                        test_items = [
                            dict(
                                title="",
                                text="",
                                img="https://github.com/naman19436/pixgen-ai/blob/main/gallery/black.png?raw=true",
                            ),
                            dict(
                                title="",
                                text="",
                                img="https://github.com/naman19436/pixgen-ai/blob/main/gallery/white.png?raw=true",
                            ),
                            ]
                        if image:
                            st.toast(
                                ':gray[Your image has been generated!]', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = image

                            # Displaying the image
                            col1, col2= st.columns([1,1])
                            # for i in get_monitors():
                            #     if i.is_primary: 
                            #         width = int(i.width*0.4)
                            
                            with col1:
                                st.image(all_images[0],width=700, use_column_width = "always")
                            with col2:
                                with st.container():
                                    st.image(all_images[1],width=400)
                                with st.container():
                                    st.image(all_images[2],width=400)
                                # carousel(items=test_items, width = 0.6)
                            
                                
                        st.session_state.all_images = all_images

                status.update(label="",
                              state="complete", expanded=False)
            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")

    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    with gallery_placeholder.container():
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            st.image("gallery/sample1.png")
        with col2:
            st.image("gallery/sample2.jpg")
        with col3:
            st.image("gallery/sample4.png")
        with col4:
            st.image("gallery/sample3.png")
        # img = image_select(
        #     label="",
        #     images=[
        #         "gallery/sample1.png", "gallery/sample2.jpg",
        #         "gallery/sample12.png", "gallery/sample2W.png",
        #     ],
        #     use_container_width=True,
        # )
        # image = Image.open("C:/Users/Ishan/Desktop/PixelParadise/pixgen-ai/gallery/farmer_sunset.png")
        # bytes = image.tobytes()
        # mystr = base64.b64encode(bytes)
        # carousel_items = [
        #                     dict(
        #                         title="Slide 1",
        #                         text="White Tshirt",
        #                         img="https://github.com/naman19436/pixgen-ai/blob/main/gallery/cheetah.png?raw=true"
        #                     ),
        #                     dict(
        #                         title="Slide 2",
        #                         text="Black Tshirt",
        #                         img = "../../../../../../../../../C:/Users/Ishan/Desktop/PixelParadise/pixgen-ai/gallery/farmer_sunset.png"
        #                     ),
        #                 ]
        # carousel(items=carousel_items, width=0.5)


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
        st.text("")

    # test_items = [
    #     dict(
    #         title="Slide 1",
    #         text="A tree in the savannah",
    #         img=r"C:/Users/Ishan/Desktop/PixelParadise/pixgen-ai/gallery/farmer_sunset.png",
    #         link="https://discuss.streamlit.io/t/new-component-react-bootstrap-carousel/46819"
    #     ),
    #     dict(
    #         title="Slide 2",
    #         text="A wooden bridge in a forest in Autumn",
    #         img="https://img.freepik.com/free-photo/beautiful-wooden-pathway-going-breathtaking-colorful-trees-forest_181624-5840.jpg?w=1380&t=st=1688825780~exp=1688826380~hmac=dbaa75d8743e501f20f0e820fa77f9e377ec5d558d06635bd3f1f08443bdb2c1",
    #         link="https://github.com/thomasbs17/streamlit-contributions/tree/master/bootstrap_carousel"
    #     ),
    #     dict(
    #         title="Slide 3",
    #         text="A distant mountain chain preceded by a sea",
    #         img="https://img.freepik.com/free-photo/aerial-beautiful-shot-seashore-with-hills-background-sunset_181624-24143.jpg?w=1380&t=st=1688825798~exp=1688826398~hmac=f623f88d5ece83600dac7e6af29a0230d06619f7305745db387481a4bb5874a0",
    #         link="https://github.com/thomasbs17/streamlit-contributions/tree/master"
    #     ),
    #     ]

    # carousel(items=test_items, width=1)
    
    with form_placeholder.container():
        submitted, prompt = configure_searchbar()
        negative_prompt = ""
        main_page(submitted, prompt, negative_prompt)

if __name__ == "__main__":
    main()
