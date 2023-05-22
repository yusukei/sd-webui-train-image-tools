import launch
import importlib

packages = []
_packages = [
    "onnx",
    "onnxruntime-gpu",
    "opencv-python",
    "numpy",
    "Pillow",
    "dlib-binary",
    "rembg",
    "pooch",
]

for name in _packages:
    try:
        importlib.__import__(name)
    except ImportError:
        packages.append(name)

for package in packages:
    if not launch.is_installed(package):
        launch.run_pip(f'install {package}', desc=f'{package} for Train Image Tools')

