import modules.scripts as scripts
import gradio as gr
import os
import tempfile

from modules import script_callbacks
import modules.shared as shared

from PIL import Image

import numpy as np
import dlib
import cv2

from scripts.rembg_interface import remove


seg_models = ['u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta']

def pil2cv(image: Image) -> np.ndarray:
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGR)
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def save_image_dir(image, path, basename, extension='png'):
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Generate the filename
    filename = f"{basename}.{extension}"
    full_path = os.path.join(path, filename)

    # Save the image
    image.save(full_path)

    return full_path


def remove_bg(image: Image, model: str, is_cpu_only: bool) -> Image:
    return remove(image, model, is_cpu_only)


def trimming_face(image: Image, padding: float) -> list[Image]:
    # convert to OpenCV format
    image = pil2cv(image)

    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(image, 1)

    results = []
    for i in range(0, len(faces)):
        img_h, img_w, c = image.shape
        face_h = int(faces[i].bottom() - faces[i].top())
        face_w = int(faces[i].right() - faces[i].left())

        rect_top = int(faces[i].top()) - (face_h * padding)
        if rect_top < 0:
            rect_top = 0
        rect_bottom = int(faces[i].bottom()) + (face_h * padding)
        if rect_bottom > img_h:
            rect_bottom = img_h
        rect_left = int(faces[i].left()) - (face_w * padding)
        if rect_left < 0:
            rect_left = 0
        rect_right = int(faces[i].right()) + (face_w * padding)
        if rect_right > img_w:
            rect_right = img_w

        face_img = image[int(rect_top):int(rect_bottom), int(rect_left):int(rect_right)]

        # convert to PIL format
        face_img = cv2pil(face_img)

        results.append(face_img)

    return results


def crop_to_square(image: Image) -> Image:
    width, height = image.size
    square_size = min(image.size)

    if width > height:
        top = 0
        bottom = square_size
        left = (width - square_size) / 2
        right = left + square_size
        box = (left, top, right, bottom)
    else:
        left = 0
        right = square_size
        top = (height - square_size) / 2
        bottom = top + square_size
        box = (left, top, right, bottom)

    return image.crop(box)


def process_image(
        image: Image, is_remove_bg: bool, is_face_only: bool, is_crop: bool,
        padding: float, model: str, is_cpu_only: bool):
    processed = []

    if is_remove_bg:
        print("remove_bg")
        image = remove_bg(image, model, is_cpu_only)
        # image = cv2pil(pil2cv(image))

    if is_face_only:
        print("trimming_face")
        processed.extend(trimming_face(image, padding))
    else:
        processed.append(image)

    if is_crop:
        print("crop_to_square")
        tmp = []
        for img in processed:
            tmp.append(crop_to_square(img))
        processed = tmp

    return processed


def processing(single_image, input_dir, output_dir, show_result, input_tab_state, is_remove_bg, is_face_only, is_crop, padding, model, is_cpu_only):
    # 0: single
    if input_tab_state == 0:
        processed = process_image(single_image, is_remove_bg, is_face_only, is_crop, padding, model, is_cpu_only)
        return processed

    elif input_tab_state == 2:
        processed = []
        files = shared.listfiles(input_dir)
        for f in files:
            try:
                image = Image.open(f)
            except Exception:
                continue
            print(f)

            imgs = process_image(image, is_remove_bg, is_face_only, is_crop, padding, model, is_cpu_only)
            processed.extend(imgs)

            if output_dir != "":
                i = 0
                for img in imgs:
                    basename = os.path.splitext(os.path.basename(f))[0]
                    ext = os.path.splitext(f)[1][1:]
                    save_image_dir(
                        img,
                        path=output_dir,
                        basename=basename+'-%d'%(i),
                        extension="png",
                    )
                    i += 1
        if (show_result):
            return processed
        else:
            return None

# コンポーネントを作成
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        input_tab_state = gr.State(value=0)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem(label="Single") as input_tab_single:
                        single_image = gr.Image(type="pil")
                    with gr.TabItem(label="Batch from Dir") as input_tab_dir:
                        input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                        output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)
                        show_result = gr.Checkbox(label="Show result images", value=True)
                with gr.Accordion("Options", open=True):
                    is_remove_bg = gr.Checkbox(label="Remove Background", show_label=True, value=True)
                    is_face_only = gr.Checkbox(label="Face Only", show_label=True, value=True)
                    padding = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Face Padding", show_label=True)
                    is_crop = gr.Checkbox(label="Crop to Square", show_label=True, value=True)
                    model = gr.Dropdown(seg_models, label="Model", value="u2net")
                    is_cpu_only = gr.Checkbox(label="CPU Only", show_label=True, value=True)
            with gr.Column():
                gallery = gr.Gallery(label="outputs", show_label=True, elem_id="gallery").style(grid=2)
                submit = gr.Button(value="Submit")

                # 0: single 1: batch 2: batch dir
                input_tab_single.select(fn=lambda: 0, inputs=[], outputs=[input_tab_state])
                input_tab_dir.select(fn=lambda: 2, inputs=[], outputs=[input_tab_state])
                submit.click(
                    processing,
                    inputs=[single_image, input_dir, output_dir, show_result, input_tab_state,
                            is_remove_bg, is_face_only, is_crop, padding, model, is_cpu_only],
                    outputs=gallery
                )

        return [(ui_component, "Train Image Tools", "loratools")]

script_callbacks.on_ui_tabs(on_ui_tabs)
