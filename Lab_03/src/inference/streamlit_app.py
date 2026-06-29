"""Streamlit user interface for the delivered CNN model."""

from __future__ import annotations

from PIL import Image

from src.config import AppConfig
from src.data.face_aligner import FaceAligner
from src.inference.predictor import CNNPredictor
from src.utils import resolve_device


def run_app() -> None:
    """Render the upload/camera, face alignment and prediction workflow."""

    import streamlit as st

    st.set_page_config(
        page_title="UTKFace: genero y edad",
        layout="wide",
    )
    st.title("Clasificacion de genero y regresion de edad")
    st.write(
        "Ejemplo educativo con una CNN multitarea entrenada en UTKFace. "
        "La aplicacion detecta y alinea automaticamente la cara mas grande, "
        "aplicando el mismo preprocesamiento utilizado durante el entrenamiento "
        "del experimento E6 (alineacion automatica)."
    )

    config = AppConfig.from_env()
    device = resolve_device(config.device)
    st.sidebar.header("Modelo")
    st.sidebar.code(str(config.cnn_checkpoint))
    st.sidebar.write(f"Device: `{device}`")

    uploaded_file = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"],
    )
    captured_file = st.camera_input("O captura una imagen con la camara")
    source_file = captured_file if captured_file is not None else uploaded_file

    if source_file is None:
        st.info("Sube o captura una imagen para comenzar.")
        return

    image = Image.open(source_file).convert("RGB")

    aligner = FaceAligner()
    aligned_image, face_found = aligner.align_and_crop(image)

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Imagen de entrada")
        st.image(image, use_container_width=True)

    if not face_found:
        st.warning(
            "No se detecto un rostro frontal. Prueba una imagen con mejor "
            "iluminacion y una cara mas visible."
        )
        return

    with right_column:
        st.subheader("Rostro alineado y recortado")
        st.image(aligned_image, use_container_width=True)

    if not st.button("Ejecutar modelo", type="primary"):
        return

    try:
        predictor = CNNPredictor.from_checkpoint(config.cnn_checkpoint, device)
        prediction = predictor.predict(aligned_image)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(str(error))
        return

    st.subheader("Resultado")
    metric_gender, metric_age, metric_confidence = st.columns(3)
    metric_gender.metric("Genero predicho", prediction.gender_label)
    metric_age.metric("Edad estimada", f"{prediction.estimated_age:.1f} anos")
    metric_confidence.metric(
        "Confianza de genero",
        f"{prediction.gender_confidence * 100:.1f}%",
    )
