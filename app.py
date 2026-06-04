from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from photo_cleaner_core import LLMSelectionConfig, format_detection_lines, process_image


st.set_page_config(
    page_title="PhotoCleaner 全人消除系统",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def bgr_to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_to_bytes(image_bgr: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return b""
    return buffer.tobytes()


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe7;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0f766e;
            --accent-2: #ef7d57;
            --line: rgba(31, 41, 55, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(239, 125, 87, 0.14), transparent 26%),
                linear-gradient(180deg, #f8f3ea 0%, #f4efe7 100%);
            color: var(--text);
        }

        .hero {
            padding: 2rem 2rem 1.6rem 2rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,255,255,0.65));
            box-shadow: 0 18px 60px rgba(31, 41, 55, 0.10);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: 0.02em;
        }

        .hero p {
            margin: 0.6rem 0 0;
            color: var(--muted);
            line-height: 1.7;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.5rem;
        }

        .feature-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 78px;
        }

        .feature-card .label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }

        .feature-card .value {
            color: var(--text);
            font-weight: 700;
            font-size: 0.98rem;
        }

        .soft-panel {
            background: rgba(255,255,255,0.70);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1rem 0.8rem;
            box-shadow: 0 10px 32px rgba(31, 41, 55, 0.08);
        }

        .section-title {
            margin: 0 0 0.75rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.7rem 1.15rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #155e75);
            color: white;
            box-shadow: 0 8px 20px rgba(15, 118, 110, 0.22);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 26px rgba(15, 118, 110, 0.28);
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.88);
            border: 1px dashed rgba(15, 118, 110, 0.35);
            border-radius: 20px;
        }

        .stTabs [role="tablist"] {
            gap: 0.3rem;
        }

        .stTabs [role="tab"] {
            border-radius: 999px;
            padding: 0.45rem 1rem;
        }

        .stDataFrame, .stTable {
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>PhotoCleaner 全人消除系统</h1>
            <p>
                只保留单张图片入口。系统会把图片交给 Qwen 复核人物编号，再把图中所有人连续修复两次并返回前端。
            </p>
            <div class="feature-grid">
                <div class="feature-card"><div class="label">输入格式</div><div class="value">PNG / JPG / JPEG</div></div>
                <div class="feature-card"><div class="label">AI 核心</div><div class="value">Qwen + YOLOv8-seg + LaMa</div></div>
                <div class="feature-card"><div class="label">处理方式</div><div class="value">单张上传</div></div>
                <div class="feature-card"><div class="label">结果输出</div><div class="value">前端展示 + 下载</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_settings() -> dict[str, object]:
    st.sidebar.markdown("### Qwen 配置")
    llm_base_url = st.sidebar.text_input("大模型接口地址", value="http://localhost:11434/v1")
    llm_model = st.sidebar.text_input("大模型名称", value="qwen2.5-vl")
    llm_api_key = st.sidebar.text_input("大模型 API Key", value="", type="password")
    st.sidebar.caption("Qwen 只负责复核人物编号，最终会删除图中所有人并重复修复两次。")

    return {
        "llm_config": LLMSelectionConfig(
            base_url=llm_base_url,
            model=llm_model,
            api_key=llm_api_key,
        ),
    }


def show_result(result, saved_paths: dict[str, Path]) -> None:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 原图")
        st.image(bgr_to_rgb(result.original_bgr), use_container_width=True)
    with col2:
        st.markdown("#### 处理后")
        st.image(bgr_to_rgb(result.cleaned_bgr), use_container_width=True)

    metrics = st.columns(3)
    metrics[0].metric("人物数量", result.subject_count)
    metrics[1].metric("耗时(秒)", f"{result.elapsed_seconds:.2f}")
    metrics[2].metric("输出文件", Path(saved_paths["output"]).name)

    st.markdown("#### 检测明细")
    lines = format_detection_lines(result.detections)
    if lines:
        for line in lines:
            st.write(line)
    else:
        st.info("未检测到人物，系统已输出原图。")

    if result.logs:
        with st.expander("处理日志", expanded=False):
            for line in result.logs:
                st.write(line)

    st.download_button(
        label="下载处理结果",
        data=image_to_bytes(result.cleaned_bgr),
        file_name=Path(saved_paths["output"]).name,
        mime="image/png",
        use_container_width=True,
    )


def handle_single_mode(settings: dict[str, object]) -> None:
    uploaded_file = st.file_uploader("上传单张图片", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if not uploaded_file:
        st.info("请先上传一张拍照场景图片。")
        return

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.image(uploaded_file, caption="待处理图片", use_container_width=True)

    with right:
        st.markdown("#### 处理说明")
        st.write("系统会把图片交给 Qwen 复核，再把图中所有人连续修复两次。")
        start_button = st.button("开始处理单张图片", use_container_width=True)

    if not start_button:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())
        output_dir = Path(tmp_dir) / "output"

        progress = st.progress(0)
        status_box = st.empty()
        status_box.write("正在读取与分析图片...")
        progress.progress(20)

        result, saved_paths = process_image(
            image_path=temp_path,
            output_dir=output_dir,
            model_path="yolov8s-seg.pt",
            llm_config=settings["llm_config"],
            save_masks=False,
        )

        status_box.write("正在保存结果并写入日志...")
        progress.progress(90)
        st.session_state["last_result"] = result
        st.session_state["last_saved_paths"] = saved_paths
        progress.progress(100)
        status_box.success("处理完成")

    show_result(result, saved_paths)


def main() -> None:
    render_css()
    render_hero()

    settings = sidebar_settings()

    st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">单张图片处理</p>', unsafe_allow_html=True)
    handle_single_mode(settings)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
