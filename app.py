from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("ppe_predictor.onnx")


@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {MODEL_PATH.resolve()}")
    return YOLO(str(MODEL_PATH), task="detect")


def run_inference(image: Image.Image, conf: float, iou: float):
    model = load_model()
    image_np = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_np,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
    )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    rows = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = result.names.get(cls_id, str(cls_id)) if isinstance(result.names, dict) else str(cls_id)
        rows.append(
            {
                "clase": label,
                "confianza": round(confidence, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    return annotated_rgb, rows


def main() -> None:
    st.set_page_config(page_title="PPE Predictor", page_icon="🦺", layout="wide")
    st.title("Detector de EPP (PPE) con Streamlit")
    st.caption("Carga una imagen o toma una foto para ejecutar deteccion con el modelo ONNX.")

    with st.sidebar:
        st.subheader("Ajustes")
        conf = st.slider("Confianza minima", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        iou = st.slider("IoU NMS", min_value=0.10, max_value=0.95, value=0.45, step=0.05)

    source = st.radio("Fuente de imagen", ["Subir imagen", "Tomar foto"], horizontal=True)

    image_bytes: bytes | None = None

    if source == "Subir imagen":
        uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    else:
        camera_file = st.camera_input("Toma una foto")
        if camera_file is not None:
            image_bytes = camera_file.getvalue()

    if image_bytes is None:
        st.info("Esperando imagen para analizar.")
        return

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    if st.button("Ejecutar deteccion", type="primary"):
        try:
            annotated, detections = run_inference(image, conf=conf, iou=iou)
        except Exception as exc:
            st.error(f"No fue posible ejecutar la inferencia: {exc}")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detecciones")
            st.image(annotated, use_container_width=True)

        st.markdown("### Resultado estructurado")
        if detections:
            st.dataframe(detections, use_container_width=True, hide_index=True)
        else:
            st.success("No se detectaron objetos con los umbrales actuales.")


if __name__ == "__main__":
    main()
