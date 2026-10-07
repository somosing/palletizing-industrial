"""Explicit renderer selection. Physics always uses Bullet's CPU backend."""

import importlib.util
import logging


def configure_renderer(p, name, headless):
    p.setRealTimeSimulation(0)
    if name == "tiny":
        logging.info("Camera: CPU TinyRenderer; physics: CPU")
        return p.ER_TINY_RENDERER
    if name == "opengl":
        if headless:
            raise ValueError(
                "--renderer opengl requires the GUI; use egl for headless OpenGL"
            )
    elif name == "egl":
        if not headless:
            raise ValueError("--renderer egl requires --headless")
        spec = importlib.util.find_spec("eglRenderer")
        plugin = (
            p.loadPlugin(spec.origin, "_eglRendererPlugin")
            if spec and spec.origin
            else p.loadPlugin("eglRendererPlugin")
        )
        if plugin < 0:
            raise RuntimeError(
                "EGL plugin unavailable. Use --renderer tiny or the local GUI with --renderer opengl."
            )
    else:
        raise ValueError("Unknown renderer: " + name)
    logging.info(
        "Camera: hardware OpenGL requested; inspect GL_VENDOR/GL_RENDERER to confirm NVIDIA. Physics: CPU."
    )
    return p.ER_BULLET_HARDWARE_OPENGL
