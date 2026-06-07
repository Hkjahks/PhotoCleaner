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

from db import insert_record

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
    
    # 过滤掉超出 masks 范围的索引
    valid_mask_count = masks.shape[0]
    indices = indices[indices < valid_mask_count]

    union_mask = np.zeros(masks.shape[1:], dtype=bool)
    for idx in indices:
        union_mask |= masks[idx] > 0.5
    return union_mask


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def select_main_subjects(
    detections: list[dict[str, object]],
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """基于面积、位置、置信度评分识别主要人物（保留）和路人（消除）。

    评分权重：面积 40% + 中心偏离 35% + 置信度 25%
    候选人取最高分 75% 内，再从候选人中按面积筛选。
    """
    n = len(detections)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n == 1:
        return np.array([0]), np.array([], dtype=int)

    image_area = image_h * image_w
    center_x, center_y = image_w / 2, image_h / 2

    scores = []
    for det in detections:
        box = det["box"]
        area = det["area"]
        conf = det["conf"]

        area_ratio = area / image_area
        area_score = min(area_ratio / 0.05, 1.0) * 40

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        center_score = (1 - dist / max_dist) * 35

        conf_score = conf * 25
        scores.append(area_score + center_score + conf_score)

    scores_arr = np.array(scores)
    threshold = scores_arr.max() * 0.75
    candidates = np.where(scores_arr >= threshold)[0]

    if len(candidates) <= 1:
        subject_indices = (
            candidates if len(candidates) == 1 else np.array([scores_arr.argmax()])
        )
    else:
        sorted_idx = np.argsort([detections[i]["area"] for i in candidates])[::-1]
        top_area = detections[candidates[sorted_idx[0]]]["area"]
        subject_indices = np.array([
            candidates[i] for i in sorted_idx
            if detections[candidates[i]]["area"] >= top_area * 0.6
        ])

    if len(subject_indices) == 0:
        subject_indices = np.array([scores_arr.argmax()])

    stray_indices = np.setdiff1d(np.arange(n), subject_indices)
    return subject_indices.astype(int), stray_indices.astype(int)


def _tile_detect(
    model: YOLO,
    image_bgr: np.ndarray,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
    conf: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """重叠切片检测：将大图切为小块分别推理，再合并映射回原图坐标。

    小目标在全图缩放后会变得极小（<10px），YOLO 难以识别。
    切片让每个小块以接近原生分辨率推理，小块内的人保持可检测尺寸。
    """
    h, w = image_bgr.shape[:2]
    stride = int(tile_size * (1 - overlap_ratio))

    # 图太小不需要切片
    if w <= tile_size and h <= tile_size:
        return np.empty((0, 4)), np.empty((0,)), np.array([])

    all_xyxy = []
    all_conf = []
    all_masks = []

    y_starts = list(range(0, max(h - tile_size, 1), stride))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))
    x_starts = list(range(0, max(w - tile_size, 1), stride))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    for y1 in y_starts:
        for x1 in x_starts:
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            tile = image_bgr[y1:y2, x1:x2]

            # 补齐到 tile_size
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                tile_padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = tile_padded

            tile_results = model(tile, classes=[0], conf=conf, imgsz=tile_size, verbose=False)
            if not tile_results or tile_results[0].boxes is None:
                continue

            t_boxes = tile_results[0].boxes
            t_xyxy = t_boxes.xyxy.cpu().numpy()
            t_conf = t_boxes.conf.cpu().numpy()
            t_masks = (
                tile_results[0].masks.data.cpu().numpy()
                if tile_results[0].masks is not None
                else np.array([])
            )

            if len(t_xyxy) == 0:
                continue

            # 坐标映射回原图
            t_xyxy[:, [0, 2]] += x1
            t_xyxy[:, [1, 3]] += y1

            all_xyxy.append(t_xyxy)
            all_conf.append(t_conf)
            if len(t_masks) > 0:
                for m in t_masks:
                    full = np.zeros((h, w), dtype=np.float32)
                    # 取 tile 实际区域
                    m_resized = cv2.resize(m.astype(np.float32), (x2 - x1, y2 - y1))
                    full[y1:y2, x1:x2] = m_resized
                    all_masks.append(full)

    if not all_xyxy:
        return np.empty((0, 4)), np.empty((0,)), np.array([])

    merged_xyxy = np.concatenate(all_xyxy, axis=0)
    merged_conf = np.concatenate(all_conf, axis=0)
    merged_masks = np.array(all_masks) if all_masks else np.array([])

    # NMS 去重：同一个人可能被多个重叠切片检测到
    keep = _nms(merged_xyxy, merged_conf, iou_threshold=0.45)
    return merged_xyxy[keep], merged_conf[keep], (
        merged_masks[keep] if len(merged_masks) > 0 else np.array([])
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """简单的非极大值抑制，按得分降序保留不重叠的框。"""
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i_idx, i in enumerate(order):
        if suppressed[i_idx]:
            continue
        keep.append(i)
        for j_idx in range(i_idx + 1, len(order)):
            if suppressed[j_idx]:
                continue
            if compute_iou(boxes[i], boxes[order[j_idx]]) > iou_threshold:
                suppressed[j_idx] = True
    return np.array(keep, dtype=int)


@lru_cache(maxsize=1)
def load_lama_model():
    try:
        simple_lama_module = import_module("simple_lama_inpainting")
    except ImportError as exc:
        raise ImportError(
            "LaMa 后端不可用，请先安装 simple-lama-inpainting"
        ) from exc

    return simple_lama_module.SimpleLama()


def _get_inpaint_device() -> tuple[object, str]:
    """检测最佳推理设备：CUDA(NVIDIA) > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


@lru_cache(maxsize=1)
def load_sd_pipeline():
    """加载 Stable Diffusion Inpainting 管线（首次调用时自动下载模型约 2.5GB）。

    优先使用 CUDA GPU，无 CUDA 时使用 CPU（速度较慢但质量相同）。
    """
    from diffusers import StableDiffusionInpaintPipeline

    device, device_type = _get_inpaint_device()
    dtype = torch.float16 if device_type == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if device_type == "cuda":
        pipe.enable_attention_slicing()
    else:
        # CPU 模式：减少步数以缩短时间
        pipe.set_progress_bar_config(disable=True)
    return pipe, device_type


def _inpaint_sd(
    image_bgr: np.ndarray,
    person_mask: np.ndarray,
    guidance_scale: float = 7.5,
) -> np.ndarray:
    """用 Stable Diffusion Inpainting 消除人物。"""
    pipe, device_type = load_sd_pipeline()

    # CPU 用较少步数，GPU 用足量步数
    num_steps = 15 if device_type == "cpu" else 25

    h, w = image_bgr.shape[:2]
    target_size = 512
    scale = target_size / max(h, w)
    if scale < 1.0:
        new_h = (int(h * scale) // 8) * 8
        new_w = (int(w * scale) // 8) * 8
    else:
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)

    if person_mask.shape[:2] != (new_h, new_w):
        mask_resized = cv2.resize(
            person_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        mask_resized = person_mask.astype(np.float32)
    mask_pil = Image.fromarray((mask_resized > 0.5).astype(np.uint8) * 255, mode="L")

    result = pipe(
        prompt="empty clean background, no people, photorealistic, high quality, natural scene",
        negative_prompt="person, people, human, face, body, hands, legs, head, figure, man, woman, child",
        image=image_pil,
        mask_image=mask_pil,
        strength=1.0,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
    ).images[0]

    result_rgb = np.array(result.resize((w, h), Image.LANCZOS).convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


_SD_AVAILABLE = None


def _sd_is_available() -> bool:
    """检测 SD Inpainting 是否可用（只检查一次）。"""
    global _SD_AVAILABLE
    if _SD_AVAILABLE is not None:
        return _SD_AVAILABLE
    try:
        from diffusers import StableDiffusionInpaintPipeline  # noqa: F401
        _SD_AVAILABLE = True
    except ImportError:
        _SD_AVAILABLE = False
    return _SD_AVAILABLE


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


def refine_person_mask(mask: np.ndarray, image_shape: tuple[int, int], dilation_pixels: int = 4) -> np.ndarray:
    """膨胀掩码使其完整覆盖人物边缘（头发、衣物边界等）。

    dilation_pixels: 膨胀像素数，相对于约 1000px 对角线的基准。
    实际膨胀量会按图像对角线等比缩放。
    """
    if mask.size == 0:
        return np.zeros(image_shape[:2], dtype=np.uint8)

    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return mask_u8

    diag = np.sqrt(h * h + w * w)
    kernel_px = max(1, int(dilation_pixels * diag / 1000))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_px * 2 + 1, kernel_px * 2 + 1)
    )
    return cv2.dilate(mask_u8, kernel)


def composite_subjects(
    clean_background: np.ndarray,
    original_image: np.ndarray,
    subject_mask: np.ndarray,
    feather_px: int = 3,
) -> np.ndarray:
    """将原图中的主体人物抠出，贴到干净背景上，边缘羽化平滑过渡。"""
    h, w = original_image.shape[:2]

    # 统一所有输入尺寸到原图
    if clean_background.shape[:2] != (h, w):
        clean_background = cv2.resize(clean_background, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if subject_mask.shape[:2] != (h, w):
        subject_mask = cv2.resize(
            subject_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )

    mask_bool = subject_mask > 0.5
    if np.count_nonzero(mask_bool) == 0:
        return clean_background.copy()

    # 膨胀 mask 形成羽化过渡带
    ksize = max(3, feather_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
    feather_ring = mask_dilated & ~mask_bool

    if np.count_nonzero(feather_ring) == 0:
        result = clean_background.copy()
        result[mask_bool] = original_image[mask_bool]
        return result

    ring_float = feather_ring.astype(np.float32)
    blur_ksize = max(3, int(np.sqrt(h * h + w * w) / 150))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    alpha = cv2.GaussianBlur(ring_float, (blur_ksize, blur_ksize), 0)
    alpha = np.clip(alpha, 0, 1)
    full_alpha = np.where(mask_bool, 1.0, alpha)
    alpha_3ch = np.dstack([full_alpha] * 3)

    bg_f = clean_background.astype(np.float32)
    orig_f = original_image.astype(np.float32)
    return (orig_f * alpha_3ch + bg_f * (1 - alpha_3ch)).astype(np.uint8)


def remove_people_twice(image_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """消除人物：小图用 SD（≤768px），大图用 LaMa（保原生分辨率）。"""
    if person_mask.size == 0:
        return image_bgr.copy()

    shape = image_bgr.shape[:2]
    h, w = shape

    # SD 需要缩放到 512px 处理，大图缩放后会变糊，仅小图使用
    if _sd_is_available() and max(h, w) <= 768:
        try:
            mask_dilated = refine_person_mask(person_mask, shape, dilation_pixels=4)
            cleaned = _inpaint_sd(image_bgr, mask_dilated)
            cleaned = remove_stray_people_lama(cleaned, mask_dilated)
            return cleaned
        except Exception:
            pass  # SD 失败则回退 LaMa

    # LaMa：原生分辨率，适合大图
    cleaned = remove_stray_people_lama(image_bgr, person_mask)
    mask_edge = refine_person_mask(person_mask, shape, dilation_pixels=2)
    cleaned = remove_stray_people_lama(cleaned, mask_edge)
    return cleaned


@lru_cache(maxsize=1)
def load_model(model_path: str = "yolov8s-seg.pt") -> YOLO:
    return YOLO(model_path)


def analyze_image(
    image_path: str | Path,
    model_path: str = "yolov8s-seg.pt",
    min_area_ratio: float = 0.01,
) -> ProcessResult:
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
    # imgsz=1280 提升推理分辨率，让小目标有更多像素参与检测
    results = model(str(image_path), classes=[0], conf=0.02, imgsz=1280, verbose=False)

    if not results:
        raise RuntimeError("模型没有返回任何结果")

    res = results[0]
    image_h, image_w = res.orig_shape
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,), dtype=np.float32)
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

    # 多尺度检测：逐级放大图片检测微小/远处人物
    original_count = len(xyxy)
    if image_h * image_w > 50000:
        # 多级缩放配置：(放大倍数, 置信度阈值)
        scale_configs = [
            (2, 0.08),   # 2x：检测中小尺寸人物
            (3, 0.03),   # 3x：检测极小/远处人物（背面等特征不明显的）
        ]
        max_dim = 4000  # 限制放大后的最大尺寸，避免 OOM

        for scale_factor, scale_conf in scale_configs:
            try:
                scaled_h = int(image_h * scale_factor)
                scaled_w = int(image_w * scale_factor)
                # 超出最大尺寸则等比缩小
                if max(scaled_h, scaled_w) > max_dim:
                    ratio = max_dim / max(scaled_h, scaled_w)
                    scaled_h = int(scaled_h * ratio)
                    scaled_w = int(scaled_w * ratio)

                scaled_image = cv2.resize(original_bgr, (scaled_w, scaled_h))
                scaled_results = model(scaled_image, classes=[0], conf=scale_conf, imgsz=1280, verbose=False)

                if scaled_results and scaled_results[0].boxes is not None:
                    s_boxes = scaled_results[0].boxes
                    s_xyxy = s_boxes.xyxy.cpu().numpy()
                    s_conf = s_boxes.conf.cpu().numpy()
                    s_masks = (
                        scaled_results[0].masks.data.cpu().numpy()
                        if scaled_results[0].masks is not None
                        else np.array([])
                    )

                    if len(s_xyxy) == 0:
                        continue

                    # 缩放坐标映射回原图尺寸
                    scale_x = image_w / scaled_w
                    scale_y = image_h / scaled_h
                    s_xyxy_scaled = s_xyxy.copy()
                    s_xyxy_scaled[:, [0, 2]] *= scale_x
                    s_xyxy_scaled[:, [1, 3]] *= scale_y

                    # 缩放 masks 回原图尺寸
                    resized_masks = (
                        np.array([
                            cv2.resize(m.astype(np.float32), (image_w, image_h))
                            for m in s_masks
                        ])
                        if len(s_masks) > 0
                        else np.array([])
                    )

                    # 去重：只保留此尺度中新发现的人（与已有结果 IoU < 0.1）
                    iou_threshold = 0.1
                    new_mask_indices = []
                    for i, new_box in enumerate(s_xyxy_scaled):
                        is_duplicate = False
                        for old_box in xyxy:
                            if compute_iou(new_box, old_box) > iou_threshold:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            new_mask_indices.append(i)

                    if new_mask_indices:
                        new_xyxy = s_xyxy_scaled[new_mask_indices]
                        new_conf = s_conf[new_mask_indices]
                        new_masks = (
                            resized_masks[new_mask_indices]
                            if len(resized_masks) > 0
                            else np.array([])
                        )
                        xyxy = np.concatenate([xyxy, new_xyxy], axis=0)
                        conf = np.concatenate([conf, new_conf], axis=0)
                        masks = np.concatenate([masks, new_masks], axis=0)
            except Exception:
                pass  # 单级放大失败不影响其他级别

    # 切片检测：针对大图中 <20px 的极小人物，切成小块各自推理
    # 触发条件：图片任一维度超过 1280px（全图缩放后小目标丢失严重）
    if max(image_h, image_w) > 1280:
        try:
            t_xyxy, t_conf, t_masks = _tile_detect(
                model, original_bgr, tile_size=640, overlap_ratio=0.2, conf=0.05
            )
            if len(t_xyxy) > 0:
                # 去重：只保留切片中与已有结果不重叠的新发现
                iou_threshold = 0.1
                new_idx = []
                for i, tile_box in enumerate(t_xyxy):
                    is_dup = False
                    for old_box in xyxy:
                        if compute_iou(tile_box, old_box) > iou_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        new_idx.append(i)
                if new_idx:
                    xyxy = np.concatenate([xyxy, t_xyxy[new_idx]], axis=0)
                    conf = np.concatenate([conf, t_conf[new_idx]], axis=0)
                    if len(t_masks) > 0:
                        masks = np.concatenate([masks, t_masks[new_idx]], axis=0)
        except Exception:
            pass  # 切片失败不影响主流程


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

    # 构建检测信息列表供评分使用
    detections = [
        {
            "index": i,
            "box": xyxy[i].tolist(),
            "conf": float(conf[i]),
            "area": float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
        }
        for i in range(len(xyxy))
    ]

    # 规则评分：识别主要人物（保留）和路人（消除）
    subject_indices, stray_indices = select_main_subjects(detections, image_h, image_w)

    # 第一步：消除所有人，得到干净背景
    all_indices = np.arange(len(xyxy))
    all_people_mask = build_union_mask(masks, all_indices)
    clean_background = remove_people_twice(original_bgr, all_people_mask)

    # 第二步：从原图抠出主体人物，贴回干净背景
    subject_mask = build_union_mask(masks, subject_indices)
    stray_mask = build_union_mask(masks, stray_indices)

    if len(subject_indices) > 0 and len(stray_indices) > 0:
        # 有路人需要消除，同时保留主体
        cleaned_image = composite_subjects(clean_background, original_bgr, subject_mask)
    elif len(stray_indices) > 0:
        # 全是路人 → 直接输出干净背景
        cleaned_image = clean_background
    else:
        # 没有路人 → 原样输出
        cleaned_image = original_bgr.copy()

    elapsed_seconds = time.time() - start_time
    subject_index_set = set(subject_indices.tolist())
    stray_index_set = set(stray_indices.tolist())
    detections_summary = [
        DetectionSummary(
            index=i,
            box=xyxy[i].tolist(),
            conf=float(conf[i]),
            score=1.0 if i in subject_index_set else 0.0,
            area=float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
            label="subject" if i in subject_index_set else "stray",
        )
        for i in range(len(xyxy))
    ]

    logs = [
        f"检测到 {len(xyxy)} 个人物："
        f"保留 {len(subject_indices)} 个主体，消除 {len(stray_indices)} 个路人。"
    ]

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
    save_masks: bool = True,
) -> tuple[ProcessResult, dict[str, Path]]:
    start_time = time.time()
    try:
        result = analyze_image(
            image_path=image_path,
            model_path=model_path,
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
            f"#{item.index} {item.label} | conf={item.conf:.3f} | removed={int(item.score > 0)} | area={item.area:.0f}"
        )
    return lines


def run_cli(
    input_path: str,
    output_dir: str,
    model_path: str = "yolov8s-seg.pt",
) -> None:
    path = Path(input_path)
    if path.is_dir():
        image_paths = collect_images(path)
        batch_results = process_batch(
            image_paths,
            output_dir=output_dir,
            model_path=model_path,
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
        )
        print(f"处理完成: {input_path}")
        print(f"输出路径: {saved_paths['output']}")
        print(f"主体数量: {result.subject_count} | 路人数量: {result.stray_count}")
        for line in format_detection_lines(result.detections):
            print(line)
