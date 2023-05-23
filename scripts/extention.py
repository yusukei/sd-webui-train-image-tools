import os

import cv2
import dlib
import gradio as gr
import modules.shared as shared
import numpy as np
import onnxruntime as ort
import urllib.request
from PIL import Image
from modules import script_callbacks
from rembg import remove
from rembg import session_factory

bg_types = ['transparent', 'white', 'black']
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


def cv2pil(image: np.array) -> Image:
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


def save_image_dir(image: Image, path: str, basename: str, extension='png') -> str:
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Generate the filename
    filename = f"{basename}.{extension}"
    full_path = os.path.join(path, filename)

    # Save the image
    image.save(full_path)

    return full_path


def remove_bg(image: np.array, bg_type: str, model: str, is_cpu_only: bool):
    def get_available_providers():
        # cuda, cuDDNなどが正しく設定されていないとエラーになるので、
        # 強制的にCPUのみを返すようにする
        return (['CPUExecutionProvider'])

    if is_cpu_only:
        _get_available_providers = ort.get_available_providers
        ort.get_available_providers = get_available_providers

    bgcolor = (0, 0, 0, 0)
    if bg_type == 'white':
        bgcolor = (255, 255, 255, 255)
    elif bg_type == 'black':
        bgcolor = (0, 0, 0, 255)

    session = session_factory.new_session(model)
    _image = remove(image, session=session, bgcolor=bgcolor)

    if is_cpu_only:
        ort.get_available_providers = _get_available_providers

    return _image


def get_crop_size(image: Image, x, y, width, height, padding):
    img_h, img_w, c = image.shape

    top = y - (height * padding)
    top = max(0, top)

    bottom = (top + height) + (height * padding)
    bottom = min(img_h, bottom)

    left = x - (width * padding)
    left = max(0, left)

    right = (x + width) + (width * padding)
    right = min(img_w, right)

    # print(f'{x}, {y}, {x+width}, {y+height} -> {left}, {top}, {right}, {bottom}')
    return (left, top, right, bottom)


def trimming_face_photo(_image: Image, padding: float) -> list[Image]:
    image = pil2cv(_image)

    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(image, 1)

    results = []
    for i in range(0, len(faces)):
        face = faces[i]
        x = face.left()
        y = face.top()
        width = face.right() - face.left()
        height = face.bottom() - face.top()

        rect = get_crop_size(image, x, y, width, height, padding)

        face_img = _image.crop(rect)

        results.append(face_img)

    return results


def trimming_face_anime(_image: Image, padding: float) -> list[Image]:
    basedir = os.path.dirname(__file__)
    cascade_file = os.path.join(basedir, 'lbpcascade_animeface.xml')
    url = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
    if not os.path.exists('lbpcascade_animeface.xml'):
        with open(cascade_file, 'w') as fp:
            body = urllib.request.urlopen(url).read().decode('utf-8')
            fp.write(body)

    cascade = cv2.CascadeClassifier(cascade_file)

    image = pil2cv(_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                 # detector options
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(24, 24))

    results = []
    for (x, y, w, h) in faces:
        rect = get_crop_size(image, x, y, w, h, padding)
        face_img = _image.crop(rect)

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
        image: Image, is_remove_bg: bool, bg_type: str, is_face_only: bool, face_type: int, is_crop: bool,
        padding: float, model: str, is_cpu_only: bool):
    processed = []

    if is_remove_bg:
        _image = remove_bg(image, bg_type, model, is_cpu_only)
    else:
        _image = image

    if is_face_only:
        if face_type == 'Photo':
            processed.extend(trimming_face_photo(_image, padding))
        elif face_type == 'Anime':
            processed.extend(trimming_face_anime(_image, padding))
    else:
        processed.append(_image)

    if is_crop:
        tmp = []
        for img in processed:
            tmp.append(crop_to_square(img))
        processed = tmp

    return processed


def processing(single_image: Image, input_dir: str, output_dir: str, show_result: bool,
               input_tab_state: int, is_remove_bg: bool, bg_type: str, is_face_only: bool, face_type: int, is_crop: bool,
               padding: float, model: str, is_cpu_only: bool):
    # 0: single
    if input_tab_state == 0:
        processed = process_image(single_image, is_remove_bg, bg_type, is_face_only, face_type, is_crop, padding, model, is_cpu_only)
        return processed

    elif input_tab_state == 2:
        processed = []
        files = shared.listfiles(input_dir)
        count = 1
        size = len(files)

        for f in files:
            try:
                image = Image.open(f)
            except Exception:
                continue
            print(f'{count}/{size} {f}')

            imgs = process_image(image, is_remove_bg, is_face_only, face_type, is_crop, padding, model, is_cpu_only)
            processed.extend(imgs)

            if output_dir != "":
                i = 0
                for img in imgs:
                    basename = os.path.splitext(os.path.basename(f))[0]
                    ext = os.path.splitext(f)[1][1:]
                    save_image_dir(
                        img,
                        path=output_dir,
                        basename=basename + '-%d' % (i),
                        extension="png",
                    )
                    i += 1

            count += 1

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
                with gr.Accordion("Remove Background", open=True):
                    is_remove_bg = gr.Checkbox(label="Enable", show_label=True, value=True)
                    bg_type = gr.Dropdown(bg_types, label='Background Color', show_label=True, value='transparent')
                    model = gr.Dropdown(seg_models, label="Model", value="u2net")
                    is_cpu_only = gr.Checkbox(label="CPU Only", show_label=True, value=True)
                with gr.Accordion("Crop a face", open=True):
                    is_face_only = gr.Checkbox(label="Enable", show_label=True, value=True)
                    face_type = gr.Radio(['Photo', 'Anime'], label='Face Type', value='Photo')
                    padding = gr.Slider(0.0, 2.0, value=0.8, step=0.1, label="Face Padding", show_label=True)
                with gr.Accordion("Crop to Square", open=True):
                    is_crop = gr.Checkbox(label="Enable", show_label=True, value=True)
            with gr.Column():
                gallery = gr.Gallery(label="outputs", show_label=True, elem_id="gallery").style(grid=2)
                submit = gr.Button(value="Submit")

                # 0: single 1: batch 2: batch dir
                input_tab_single.select(fn=lambda: 0, inputs=[], outputs=[input_tab_state])
                input_tab_dir.select(fn=lambda: 2, inputs=[], outputs=[input_tab_state])
                submit.click(
                    processing,
                    inputs=[single_image, input_dir, output_dir, show_result, input_tab_state,
                            is_remove_bg, bg_type, is_face_only, face_type, is_crop, padding, model, is_cpu_only],
                    outputs=gallery
                )

        return [(ui_component, "Train Image Tools", "loratools")]


script_callbacks.on_ui_tabs(on_ui_tabs)
