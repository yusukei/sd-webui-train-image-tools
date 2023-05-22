
import numpy as np
from rembg import remove as remove_bg
import onnxruntime as ort


def get_session(model: str, is_cpu_only: bool):
    session = None
    try:
        # old style
        import rembg.session_base

        if is_cpu_only:
            from . import cpu_session_factory
            session = cpu_session_factory.new_session(model)
        else:
            from rembg import session_factory
            session = session_factory.new_session(model)
    except ImportError:
        # new style
        from rembg import session_factory

        session = session_factory.new_session(model)
        if is_cpu_only:
            sess_opts = ort.SessionOptions()
            session.inner_session = ort.InferenceSession(
                str(session.__class__.download_models()),
                providers=["CPUExecutionProvider"],
                sess_options=sess_opts,
            )

    return session




def remove(image: np.array, model: str="u2net", is_cpu_only: bool=False):
    session = get_session(model, is_cpu_only)
    return remove_bg(image, session)



