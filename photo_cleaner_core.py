from __future__ import annotations

# 强制使用 CPU，必须在导入 torch 相关库之前设置
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
from importlib import import_module
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

# 在导入 ultralytics 之前，先强制 torch 使用 CPU
import torch
torch.cuda.is_available = lambda: False  # 强制覆盖，让所有 GPU 检查返回 False

# 补丁 torch.jit.load，强制使用 CPU
_original_jit_load = torch.jit.load
def _patched_jit_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_jit_load(*args, **kwargs)
torch.jit.load = _patched_jit_load

from ultralytics import YOLO

from db import init_db, insert_record
from llm_subject_selector import LLMSelectionConfig, select_subject_indices

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class DetectionSummary:
    index: int
    box: list[float]
    conf: float
    score: float
    area: float
    label: str


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    subject_count: int
    stray_count: int
    elapsed_seconds: float
    original_bgr: np.ndarray
    cleaned_bgr: np.ndarray
    subject_mask: np.ndarray
    stray_mask: np.ndarray
    detections: list[DetectionSummary]
    logs: list[str]
    status: str = "success"
    error_message: str = ""


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def build_union_mask(masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if masks.size == 0 or len(indices) == 0:
        return np.zeros((0, 0), dtype=bool)

    union_mask = np.zeros(masks.shape[1:], dtype=bool)
    for idx in indices:
        union_mask |= masks[idx] > 0.5
    return union_mask


@lru_cache(maxsize=1)
def load_lama_model():
    try:
        simple_lama_module = import_module("simple_lama_inpainting")
    except ImportError as exc:
        raise ImportError(
            "LaMa 后端不可用，请先安装 simple-lama-inpainting"
        ) from exc

    return simple_lama_module.SimpleLama()


def remove_stray_people_lama(image_bgr: np.ndarray, stray_mask: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr

    if stray_mask.size == 0:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    if stray_mask.shape[:2] != (image_h, image_w):
        stray_mask = cv2.resize(
            stray_mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = (stray_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return image_bgr.copy()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask_u8, mode="L")

    lama = load_lama_model()
    result_pil = lama(image_pil, mask_pil)
    result_rgb = np.array(result_pil.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def remove_stray_people(
    image_bgr: np.ndarray,
    stray_mask: np.ndarray,
) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr

    if stray_mask.size == 0:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    if stray_mask.shape[:2] != (image_h, image_w):
        stray_mask = cv2.resize(
            stray_mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = (stray_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return image_bgr.copy()

    return remove_stray_people_lama(image_bgr, stray_mask)


@lru_cache(maxsize=1)
def load_model(model_path: str = "yolov8s-seg.pt") -> YOLO:
    return YOLO(model_path)


def analyze_image(
    image_path: str | Path,
    model_path: str = "yolov8s-seg.pt",
    subject_score_ratio: float = 0.75,
    min_area_ratio: float = 0.01,#降低最小面积阈值 - 检测更小的远处人物
    llm_config: LLMSelectionConfig | None = None,
) -> ProcessResult:
    init_db()

    start_time = time.time()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not is_supported_image(image_path):
        raise ValueError(f"不支持的图片格式: {image_path.suffix}")

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    model = load_model(model_path)
    results = model(str(image_path), classes=[0],conf=0.05,verbose=False)#conf降低置信度阈值 - 检出更多模糊人物



    res = results[0]
    image_h, image_w = res.orig_shape
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,), dtype=np.float32)
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

    # 多尺度检测：放大图片检测小人物（如果原图没有检测到足够的人）
    if len(xyxy) < 20 and image_h * image_w > 100000:  # 如果原图检测少于3人且图片足够大
        try:
            large_h, large_w = int(image_h * 1.5), int(image_w * 1.5)
            large_image = cv2.resize(original_bgr, (large_w, large_h))
            large_results = model(large_image, classes=[0], conf=0.15, verbose=False)
            
            if large_results and large_results[0].masks is not None:
                large_boxes = large_results[0].boxes
                large_xyxy = large_boxes.xyxy.cpu().numpy()
                large_conf = large_boxes.conf.cpu().numpy()
                large_masks = large_results[0].masks.data.cpu().numpy()
                # 缩放坐标映射回原图尺寸
                scale_x = image_w / large_w
                scale_y = image_h / large_h
                large_masks = large_results[0].masks.data.cpu().numpy()
                
                resized_masks = []
                for mask in large_masks:
                    resized_mask = cv2.resize(mask, (image_w, image_h))
                    resized_masks.append(resized_mask.astype(bool))
                large_masks_resized = np.array(resized_masks)
                
                # 合并检测结果
                if len(masks) > 0:
                    masks = np.concatenate([masks, large_masks_resized], axis=0)
                else:
                    masks = large_masks_resized
        except Exception:
            pass  # 多尺度检测失败不影响主流程


            
    if not results:
        raise RuntimeError("模型没有返回任何结果")

    res = results[0]
    image_h, image_w = res.orig_shape
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,), dtype=np.float32)
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

    if len(xyxy) == 0:
        cleaned_image = original_bgr.copy()
        subject_mask = np.zeros((image_h, image_w), dtype=bool)
        stray_mask = np.zeros((image_h, image_w), dtype=bool)
        elapsed_seconds = time.time() - start_time
        result = ProcessResult(
            input_path=str(image_path),
            output_path="",
            subject_count=0,
            stray_count=0,
            elapsed_seconds=elapsed_seconds,
            original_bgr=original_bgr,
            cleaned_bgr=cleaned_image,
            subject_mask=subject_mask,
            stray_mask=stray_mask,
            detections=[],
            logs=["未检测到人物，已原样输出。"],
        )
        return result

    detections = [
        {
            "index": i,
            "box": xyxy[i].tolist(),
            "conf": float(conf[i]),
            "area": float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
        }
        for i in range(len(xyxy))
    ]

    subject_indices, selector_logs = select_subject_indices(
        image_bgr=original_bgr,
        detections=detections,
        llm_config=llm_config,
    )

    all_indices = np.arange(len(xyxy))
    stray_indices = np.setdiff1d(all_indices, subject_indices)

    subject_mask = build_union_mask(masks, subject_indices)
    stray_mask = build_union_mask(masks, stray_indices)
    cleaned_image = remove_stray_people(
        original_bgr,
        stray_mask,
    )

    elapsed_seconds = time.time() - start_time
    subject_index_set = set(subject_indices.tolist())
    detections_summary = [
        DetectionSummary(
            index=i,
            box=xyxy[i].tolist(),
            conf=float(conf[i]),
            score=1.0 if i in subject_index_set else 0.0,
            area=float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
            label="subject" if i in subject_indices else "stray",
        )
        for i in range(len(xyxy))
    ]

    logs = selector_logs + [f"检测到 {len(xyxy)} 个人物，主体 {len(subject_indices)} 个，路人 {len(stray_indices)} 个。"]

    return ProcessResult(
        input_path=str(image_path),
        output_path="",
        subject_count=len(subject_indices),
        stray_count=len(stray_indices),
        elapsed_seconds=elapsed_seconds,
        original_bgr=original_bgr,
        cleaned_bgr=cleaned_image,
        subject_mask=subject_mask,
        stray_mask=stray_mask,
        detections=detections_summary,
        logs=logs,
    )


def save_result(
    result: ProcessResult,
    output_dir: str | Path,
    save_masks: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(result.input_path)
    base_name = source_path.stem

    output_image_path = output_dir / f"{base_name}_removed.png"
    cv2.imwrite(str(output_image_path), result.cleaned_bgr)

    saved_paths: dict[str, Path] = {"output": output_image_path}

    if save_masks:
        subject_mask_path = output_dir / f"{base_name}_subject_mask.png"
        stray_mask_path = output_dir / f"{base_name}_stray_mask.png"
        if result.subject_mask.size > 0:
            cv2.imwrite(str(subject_mask_path), result.subject_mask.astype(np.uint8) * 255)
        if result.stray_mask.size > 0:
            cv2.imwrite(str(stray_mask_path), result.stray_mask.astype(np.uint8) * 255)
        saved_paths["subject_mask"] = subject_mask_path
        saved_paths["stray_mask"] = stray_mask_path

    result.output_path = str(output_image_path)
    return saved_paths


def process_image(
    image_path: str | Path,
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> tuple[ProcessResult, dict[str, Path]]:
    start_time = time.time()
    try:
        result = analyze_image(
            image_path=image_path,
            model_path=model_path,
            llm_config=llm_config,
        )
        saved_paths = save_result(result, output_dir=output_dir, save_masks=save_masks)

        try:
            insert_record(
                input_path=result.input_path,
                output_path=str(saved_paths["output"]),
                subject_count=result.subject_count,
                stray_count=result.stray_count,
                status="success",
                error_message="",
                elapsed=result.elapsed_seconds,
            )
        except Exception:
            pass

        return result, saved_paths
    except Exception as exc:
        try:
            insert_record(
                input_path=str(image_path),
                output_path="",
                subject_count=0,
                stray_count=0,
                status="failed",
                error_message=str(exc),
                elapsed=time.time() - start_time,
            )
        except Exception:
            pass
        raise


def collect_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        return []
    if path.is_file():
        return [path] if is_supported_image(path) else []
    return [item for item in sorted(path.iterdir()) if item.is_file() and is_supported_image(item)]


def process_batch(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> list[tuple[Path, ProcessResult | None, str | None]]:
    results: list[tuple[Path, ProcessResult | None, str | None]] = []
    for image_path in image_paths:
        path = Path(image_path)
        try:
            result, _ = process_image(
                image_path=path,
                output_dir=output_dir,
                model_path=model_path,
                llm_config=llm_config,
                save_masks=save_masks,
            )
            results.append((path, result, None))
        except Exception as exc:
            results.append((path, None, str(exc)))
    return results


def format_detection_lines(detections: list[DetectionSummary]) -> list[str]:
    lines: list[str] = []
    for item in detections:
        lines.append(
            f"#{item.index} {item.label} | conf={item.conf:.3f} | keep={int(item.score > 0)} | area={item.area:.0f}"
        )
    return lines


def run_cli(
    input_path: str,
    output_dir: str,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
) -> None:
    path = Path(input_path)
    if path.is_dir():
        image_paths = collect_images(path)
        batch_results = process_batch(
            image_paths,
            output_dir=output_dir,
            model_path=model_path,
            llm_config=llm_config,
        )
        print(f"共处理 {len(batch_results)} 张图片")
        for item_path, result, error in batch_results:
            if result is None:
                print(f"失败: {item_path} -> {error}")
                continue
            print(f"成功: {item_path} -> {result.output_path}")
    else:
        result, saved_paths = process_image(
            input_path,
            output_dir=output_dir,
            model_path=model_path,
            llm_config=llm_config,
        )
        print(f"处理完成: {input_path}")
        print(f"输出路径: {saved_paths['output']}")
        print(f"主体数量: {result.subject_count} | 路人数量: {result.stray_count}")
        for line in format_detection_lines(result.detections):
            print(line)
