"""Optional Streamlit front-end.  Run:  streamlit run app/streamlit_app.py
Requires: pip install 'mindtext[app]'.  The core package and CLI work without it.

Shows only numbers read from real report.json files in ./reports; if no run exists it
says so instead of displaying placeholder metrics.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from mindtext import DISCLAIMER
from mindtext.errors import MindTextError
from mindtext.persistence import load_bundle
from mindtext.reporting import find_report, read_json

st.set_page_config(page_title="MindText (research demo)", layout="wide")
st.title("MindText - research demo")
st.warning(
    DISCLAIMER + " Models trained on social-media text or a small interview dataset cannot assess any individual. "
    "If you or someone you know is struggling, please contact a qualified professional or local emergency/crisis services."
)

reports_dir = st.sidebar.text_input("Reports directory", "reports")
model_path = st.sidebar.text_input("Model file (trusted source only)", "artifacts/model.joblib")

tab_over, tab_res, tab_demo = st.tabs(["Overview", "Latest experiment", "Try the classifier"])

with tab_over:
    st.markdown(
        "MindText is an educational ML pipeline for (1) multi-class classification of Reddit mental-health-community "
        "posts and (2) binary classification of PHQ-derived labels from acoustic-feature statistics. "
        "Labels describe *where a post was written* or *a questionnaire cut-off*, not a clinical diagnosis. "
        "See docs/limitations.md."
    )

with tab_res:
    try:
        rep = read_json(find_report(reports_dir))
    except MindTextError as exc:
        st.info(str(exc))
    else:
        st.caption(f"Run {rep['run_id']} - data: {rep['data']['path']} {rep['data'].get('description', '')}")
        rows = [{"model": m["name"], "provenance": m["provenance"], "val macro-F1": (m["val"] or {}).get("metrics", {}).get("macro_f1"),
                 "test accuracy": m["test"]["metrics"]["accuracy"], "test macro-F1": m["test"]["metrics"]["macro_f1"]} for m in rep["models"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write(f"Selected on validation data: **{rep['selected_model']}**")
        for w in rep.get("warnings", []):
            st.caption(f"Warning: {w}")
        for fig in rep.get("figures", []):
            fp = Path(reports_dir) / rep["run_id"] / fig
            if fp.exists():
                st.image(str(fp), caption=fig)

with tab_demo:
    try:
        bundle = load_bundle(model_path)
    except (MindTextError, OSError) as exc:
        st.info(f"No usable model loaded: {exc}")
    else:
        if bundle.task != "text":
            st.info("The loaded model is a tabular (acoustic) model; use `python -m mindtext predict --input features.csv`.")
        else:
            txt = st.text_area("Enter an example sentence (do not paste anyone's private information)")
            if st.button("Classify") and txt.strip():
                out = bundle.predict_frame([txt]).iloc[0]
                st.write(f"Predicted class: **{out['prediction']}** (score {out['score']:.3f}, {out['score_kind']}).")
                st.caption(DISCLAIMER)
