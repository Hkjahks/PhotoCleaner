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
    model: str = "gpt-4o-mini"
    api_key: str = ""
    timeout_seconds: int = 90
    request_mode: str = "auto"


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
                subject_indices = payload.get(
                    "person_indices",
                    payload.get("subject_indices", payload.get("keep_indices", [])),
                )
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


def should_use_vision(config: LLMSelectionConfig) -> bool:
    mode = config.request_mode.lower().strip()
    if mode == "vision":
        return True
    if mode == "text":
        return False

    haystack = f"{config.base_url} {config.model}".lower()
    if "deepseek" in haystack:
        return False
    return True


def build_prompt(detections: list[dict[str, object]], use_vision: bool) -> str:
    intro = (
        "请根据原图和编号框图找出图中所有人物，并返回所有人物编号。\n"
        if use_vision
        else "请根据候选人物的编号、位置、面积和置信度找出图中所有人物，并返回所有人物编号。\n"
    )

    return (
        "你是照片人物识别器。图中每个检测框都标了编号。\n"
        + intro
        + "只返回 JSON，不要输出多余文字。格式必须是："
        "{\"person_indices\":[0,1,2]}\n"
        "要求：\n"
        "1. person_indices 只允许包含整数编号。\n"
        "2. 必须返回图中所有人物对应的编号，不要漏掉任何人。\n"
        "3. 如果看不清某个编号框，也要尽量根据图像内容补全。\n"
        "4. 不要返回解释。\n"
        "候选人物信息：\n"
        + "\n".join(
            [
                f'#{item["index"]}: box={item["box"]}, conf={item["conf"]:.3f}, area={item["area"]:.0f}'
                for item in detections
            ]
        )
    )


def call_openai_compatible_vision_model(
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: list[dict[str, object]],
    config: LLMSelectionConfig,
) -> tuple[list[int], str]:
    use_vision = should_use_vision(config)
    prompt = build_prompt(detections, use_vision=use_vision)

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
                    *(
                        [
                            {"type": "image_url", "image_url": {"url": image_to_data_url(image_bgr)}},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(overlay_bgr)}},
                        ]
                        if use_vision
                        else []
                    ),
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
    except urllib_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        details = f"HTTP {exc.code} {exc.reason}"
        if error_text:
            details = f"{details}: {error_text}"
        raise RuntimeError(f"调用大模型失败: {details}") from exc
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


def llm_subject_select(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    if not detections:
        return np.array([], dtype=int), []

    if llm_config is None:
        llm_config = LLMSelectionConfig()

    overlay = render_detection_overlay(image_bgr, detections)
    subject_indices, raw_content = call_openai_compatible_vision_model(
        image_bgr=image_bgr,
        overlay_bgr=overlay,
        detections=detections,
        config=llm_config,
    )
    if not subject_indices:
        raise RuntimeError(f"大模型没有返回有效主体编号: {raw_content}")

    logs = ["Qwen 已完成人物编号识别。", f"Qwen 返回: {raw_content}"]
    return np.array(subject_indices, dtype=int), logs


def select_subject_indices(
    image_bgr: np.ndarray,
    detections: list[dict[str, object]],
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    return llm_subject_select(image_bgr=image_bgr, detections=detections, llm_config=llm_config)