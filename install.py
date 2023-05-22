import launch

packages = [
    "onnx",
    "onnxruntime-gpu",
    "opencv-python",
    "numpy",
    "Pillow",
    "dlib-binary",
    "rembg",
]

for package in packages:
    if not launch.is_installed(package):
        launch.run_pip(f'install {package}', desc=f'{package} for Train Image Tools')

