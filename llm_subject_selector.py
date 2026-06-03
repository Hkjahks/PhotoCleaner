from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np


@dataclass
class LLMSelectionConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5:9b"
    api_key: str = ""
    timeout_seconds: int = 90


def image_to_data_url(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("无法编码图片")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_detection_overlay(image_bgr: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for detection in detections:
        box = detection["box"]
        index = detection["index"]
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 3)
        label = f"#{index}"
        text_origin = (x1, max(22, y1 - 8))
        cv2.putText(
            overlay,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def parse_llm_indices(text: str, total_count: int) -> list[int]:
    text = text.strip()
    if not text:
        raise ValueError("模型没有返回内容")

    candidate_texts = [text]
    if "```" in text:
        candidate_texts.append(text.split("```", 2)[1])

    for candidate in candidate_texts:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                subject_indices = payload.get("subject_indices", payload.get("keep_indices", []))
            elif isinstance(payload, list):
                subject_indices = payload
            else:
                continue

            if not isinstance(subject_indices, list):
                continue

            cleaned = sorted({int(item) for item in subject_indices if isinstance(item, (int, float, str))})
            return [item for item in cleaned if 0 <= item < total_count]
        except Exception:
            continue

    raise ValueError(f"无法解析模型返回的主体编号: {text}")


def call_openai_compatible_vision_model(
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: list[dict[str, object]],
    config: LLMSelectionConfig,
) -> tuple[list[int], str]:
    prompt = (
        "你是照片主体选择器。图中每个检测框都标了编号。\n"
        "请根据原图和编号框图判断哪些人物应该保留，通常保留的是拍摄主体、最重要的人物、"
        "以及明确属于同一主体群体的人。\n"
        "只返回 JSON，不要输出多余文字。格式必须是："
        "{\"subject_indices\":[0,2]}\n"
        "要求：\n"
        "1. subject_indices 只允许包含整数编号。\n"
        "2. 如果只有一个主要人物，就只返回那一个编号。\n"
        "3. 如果多个检测框都属于同一组主体，可以全部返回。\n"
        "4. 不要返回解释。"
    )

    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格输出 JSON 的视觉判断助手。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_bgr)}},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(overlay_bgr)}},
                    {
                        "type": "text",
                        "text": "候选人物信息：\n"
                        + "\n".join(
                            [
                                f'#{item["index"]}: box={item["box"]}, conf={item["conf"]:.3f}, area={item["area"]:.0f}'
                                for item in detections
                            ]
                        ),
                    },
                ],
            },
        ],
    }

    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    request = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except urllib_error.URLError as exc:
        raise RuntimeError(f"调用大模型失败: {exc}") from exc

    response_json = json.loads(response_text)
    choices = response_json.get("choices", [])
    if not choices:
        raise RuntimeError("大模型没有返回 choices")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    subject_indices = parse_llm_indices(content, total_count=len(detections))
    return subject_indices, content


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

def rule_based_select(
    detections: list[dict[str, object]],
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, list[str]]:
    if not detections:
        return np.array([], dtype=int), []

    image_area = image_h * image_w
    center_x, center_y = image_w / 2, image_h / 2

    scores = []
    for det in detections:
        box = det["box"]
        conf = det["conf"]
        area = det["area"]

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

    scores = np.array(scores)
    max_score = scores.max()
    # 所有候选
    threshold = max_score * 0.75
    candidates = np.where(scores >= threshold)[0]

    if len(candidates) <= 1:

        subject_indices = candidates if len(candidates) == 1 else np.array([scores.argmax()])

    else:

        # 按面积从大到小排序

        sorted_idx = np.argsort([detections[i]["area"] for i in candidates])[::-1]

        top_area = detections[candidates[sorted_idx[0]]]["area"]


        # 保留面积达到最大值 60% 以上的框

        subject_indices = np.array([candidates[i] for i in sorted_idx

                                   if detections[candidates[i]]["area"] >= top_area * 0.6])


    if len(subject_indices) == 0:

        subject_indices = np.array([scores.argmax()])


    logs = [...]

    print(f"[DEBUG] 主体: {subject_indices.tolist()}, 分数: {[round(scores[i],1) for i in subject_indices]}")

    return subject_indices.astype(int), logs


def llm_subject_select(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    if not detections:
        return np.array([], dtype=int), []

    if llm_config is None:
        llm_config = LLMSelectionConfig()

    # 模型名或地址为空时，直接用规则
    if not llm_config.model.strip() or not llm_config.base_url.strip():
        image_h, image_w = image_bgr.shape[:2]
        return rule_based_select(detections, image_h, image_w)

    overlay = render_detection_overlay(image_bgr, detections)
    subject_indices, raw_content = call_openai_compatible_vision_model(
        image_bgr=image_bgr,
        overlay_bgr=overlay,
        detections=detections,
        config=llm_config,
    )
    if not subject_indices:
        raise RuntimeError(f"大模型没有返回有效主体编号: {raw_content}")

    logs = ["大模型已完成主体判断。", f"大模型返回: {raw_content}"]
    return np.array(subject_indices, dtype=int), logs


def select_subject_indices(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    if not detections:
        return np.array([], dtype=int), []

    if llm_config is None:
        llm_config = LLMSelectionConfig()

    # 模型名或地址为空时，直接用规则
    if not llm_config.model.strip() or not llm_config.base_url.strip():
        image_h, image_w = image_bgr.shape[:2]
        return rule_based_select(detections, image_h, image_w)

    # 尝试调用 LLM，失败则 fallback 到规则
    try:
        return llm_subject_select(image_bgr=image_bgr, detections=detections, llm_config=llm_config)
    except Exception as exc:
        image_h, image_w = image_bgr.shape[:2]
        subject_indices, logs = rule_based_select(detections, image_h, image_w)
        logs.append(f"LLM 调用失败（{type(exc).__name__}: {exc}），自动切换规则判断。")
        return subject_indices, logs

