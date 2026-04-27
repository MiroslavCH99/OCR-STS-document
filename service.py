import csv
import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR
from rapidfuzz import fuzz

from config import RulesConfig


_OCR_ENGINE: PaddleOCR | None = None


@dataclass(frozen=True)
class AnchorCandidate:
    line_indices: tuple[int, ...]
    center_idx: int
    score: float
    source: str


@dataclass(frozen=True)
class NameTokenCandidate:
    token: str
    confidence: float
    line_idx: int
    distance: int
    line_text: str


@dataclass(frozen=True)
class FioCandidate:
    surname: str | None
    name: str | None
    patronymic: str | None
    confidence: float
    anchor_score: float
    parts_count: int
    quality_score: float
    rotation: int
    ocr_variant: str


def get_ocr_engine() -> PaddleOCR:
    """Возвращает инициализированный экземпляр PaddleOCR."""
    global _OCR_ENGINE

    if _OCR_ENGINE is None:
        _OCR_ENGINE = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    return _OCR_ENGINE


def normalize_text(s: str, rules: RulesConfig) -> str:
    """Нормализует OCR-текст."""
    normalized = str(s).strip().upper()
    normalized = normalized.translate(rules["text"]["latin_to_cyr_table"])
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_for_anchor(s: str, rules: RulesConfig) -> str:
    """Нормализует строку для поиска якоря, оставляя только кириллицу и пробелы."""
    normalized = normalize_text(s, rules)
    normalized = re.sub(r"[^А-ЯЁ ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_tokens(text: str) -> tuple[str, ...]:
    """Извлекает кириллические токены из строки."""
    return tuple(re.findall(r"[А-ЯЁ]+", str(text).upper()))


def get_letter_counts(s: str) -> tuple[int, int]:
    """Считает число латинских и кириллических букв в строке."""
    text = str(s).upper()
    latin_count = len(re.findall(r"[A-Z]", text))
    cyr_count = len(re.findall(r"[А-ЯЁ]", text))
    return latin_count, cyr_count


def is_pure_latin_line(s: str) -> bool:
    """Проверяет, что строка содержит только латиницу без кириллицы."""
    latin_count, cyr_count = get_letter_counts(s)
    return latin_count > 0 and cyr_count == 0


def has_unconverted_latin_noise(s: str, rules: RulesConfig) -> bool:
    """Проверяет, что после конвертации осталось слишком много латинского шума."""
    raw = str(s).strip().upper()

    letters = re.findall(r"[A-ZА-ЯЁ]", raw)

    if not letters:
        return False

    converted = raw.translate(rules["text"]["latin_to_cyr_table"])
    unconverted_latin_count = len(re.findall(r"[A-Z]", converted))
    ratio = unconverted_latin_count / len(letters)

    return (
        unconverted_latin_count > 0
        and ratio >= rules["text"]["max_unconverted_latin_ratio"]
    )


def token_matches_stopword(token: str, stopword: str) -> bool:
    """Проверяет совпадение токена со стоп-словом с учетом нечеткого сравнения."""
    if len(stopword) <= 3:
        return token == stopword

    if token == stopword or token.startswith(stopword) or stopword in token:
        return True

    if len(token) >= 6 and len(stopword) >= 6:
        fuzzy_score = fuzz.partial_ratio(token, stopword)

        if fuzzy_score >= 85 and token[0] == stopword[0] and token[-1] == stopword[-1]:
            return True

    return False


def token_has_any_stopword(token: str, stopwords: Iterable[str]) -> bool:
    """Проверяет, совпадает ли токен с любым стоп-словом из списка."""
    for stopword in stopwords:
        if token_matches_stopword(token, stopword):
            return True

    return False


def contains_any_stopword(
    text: str, stopwords: Iterable[str], rules: RulesConfig
) -> bool:
    """Проверяет наличие хотя бы одного стоп-слова в строке после нормализации."""
    normalized = normalize_text(text, rules)

    for token in get_tokens(normalized):
        if token_has_any_stopword(token, stopwords):
            return True

    return False


def is_address_line(text: str, rules: RulesConfig) -> bool:
    """Определяет, похожа ли строка на адресную часть документа."""
    return contains_any_stopword(text, rules["filters"]["address_stop_words"], rules)


def is_service_line(text: str, rules: RulesConfig) -> bool:
    """Определяет, является ли строка служебной (не частью ФИО)."""
    return contains_any_stopword(text, rules["filters"]["service_stop_words"], rules)


def apply_suffix_corrections(token: str, rules: RulesConfig) -> str:
    """Применяет коррекции типичных OCR-ошибок в суффиксах токена."""
    corrected = token

    for bad_suffix, good_suffix in rules["postprocess"]["suffix_corrections"].items():
        if corrected.endswith(bad_suffix):
            corrected = f"{corrected[:-len(bad_suffix)]}{good_suffix}"
            break

    return corrected


def is_patronymic_token(token: str, rules: RulesConfig) -> bool:
    """Проверяет, похож ли токен на отчество по суффиксу."""
    return any(
        token.endswith(suffix) for suffix in rules["postprocess"]["patronymic_suffixes"]
    )


def is_surname_token(token: str, rules: RulesConfig) -> bool:
    """Проверяет, похож ли токен на фамилию по суффиксу."""
    return any(
        token.endswith(suffix) for suffix in rules["postprocess"]["surname_suffixes"]
    )


def get_owner_anchor_score(
    text: str, rules: RulesConfig
) -> tuple[float, bool, bool, str]:
    """Считает score строки как якоря блока владельца."""
    normalized = normalize_for_anchor(text, rules)

    if not normalized:
        return 0.0, False, False, normalized

    cyr_count = len(re.findall(r"[А-ЯЁ]", normalized))

    has_owner = "СОБСТВЕННИК" in normalized
    has_holder = "ВЛАДЕЛЕЦ" in normalized

    if cyr_count < rules["anchor"]["min_cyr_letters"] and not (has_owner or has_holder):
        return 0.0, has_owner, has_holder, normalized

    scores = [
        fuzz.partial_ratio(normalized, phrase) for phrase in rules["anchor"]["phrases"]
    ]
    scores.extend(
        fuzz.token_set_ratio(normalized, phrase)
        for phrase in rules["anchor"]["phrases"]
    )

    best_score = float(max(scores))

    if has_holder and not has_owner:
        best_score -= rules["anchor"]["weaker_keyword_penalty"]

    if has_owner:
        best_score += rules["anchor"]["owner_keyword_bonus"]

    best_score = max(0.0, min(100.0, best_score))

    return best_score, has_owner, has_holder, normalized


def is_hard_reject_token(token: str, rules: RulesConfig) -> bool:
    """Проверяет токен по списку жестких запретов (служебные/адресные фрагменты)."""
    for fragment in rules["filters"]["hard_reject_substrings"]:
        if len(fragment) <= 3:
            if token == fragment:
                return True
            continue

        if fragment in token:
            return True

    return False


def extract_name_tokens_from_line(line_text: str, rules: RulesConfig) -> list[str]:
    """Извлекает кандидаты токенов ФИО из одной OCR-строки."""
    raw = str(line_text).strip()

    if not raw:
        return []

    normalized = normalize_text(raw, rules)

    if not normalized:
        return []

    if re.search(r"\d", normalized):
        return []

    if contains_any_stopword(normalized, rules["filters"]["field_words"], rules):
        return []

    if is_address_line(normalized, rules):
        return []

    if is_service_line(normalized, rules):
        return []

    anchor_score, _, _, _ = get_owner_anchor_score(normalized, rules)

    if anchor_score >= rules["anchor"]["min_score"]:
        return []

    if is_pure_latin_line(raw):
        return []

    if has_unconverted_latin_noise(raw, rules):
        return []

    cleaned = re.sub(r"[^А-ЯЁ\- ]", " ", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return []

    raw_tokens = re.findall(r"[А-ЯЁ\-]+", cleaned)
    result: list[str] = []

    for token in raw_tokens:
        token = token.strip("-")

        if not token:
            continue

        if len(token) < rules["text"]["min_name_token_len"]:
            continue

        if len(token) > rules["text"]["max_name_token_len"]:
            continue

        token = apply_suffix_corrections(token, rules)

        if token_has_any_stopword(token, rules["all_stop_words"]):
            continue

        if is_hard_reject_token(token, rules):
            continue

        result.append(token)

    unique_tokens: list[str] = []

    for token in result:
        if token not in unique_tokens:
            unique_tokens.append(token)

    if len(unique_tokens) > rules["text"]["max_tokens_per_line"]:
        return []

    return unique_tokens


def add_unique_name_token(
    target: list[NameTokenCandidate], candidate: NameTokenCandidate
) -> None:
    """Добавляет токен в список, заменяя дубликат при необходимости."""
    for idx, existing in enumerate(target):
        if candidate.token == existing.token:
            if candidate.confidence > existing.confidence:
                target[idx] = candidate
            return

        if fuzz.ratio(candidate.token, existing.token) >= 95:
            if candidate.confidence > existing.confidence:
                target[idx] = candidate
            return

    target.append(candidate)


def score_token_bundle(
    bundle: tuple[NameTokenCandidate, ...], rules: RulesConfig
) -> float:
    """Вычисляет базовый скор набора токенов ФИО."""
    confidences = [entry.confidence for entry in bundle]
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    has_patronymic = any(is_patronymic_token(entry.token, rules) for entry in bundle)
    has_surname = any(is_surname_token(entry.token, rules) for entry in bundle)

    line_indices = [entry.line_idx for entry in bundle]
    line_span = max(line_indices) - min(line_indices) if line_indices else 0

    score = len(bundle) * rules["scoring"]["parts_weight"]
    score += mean_conf * rules["scoring"]["confidence_weight"]
    score -= line_span * rules["scoring"]["line_span_penalty"]

    if has_patronymic:
        score += rules["scoring"]["patronymic_bonus"]

    if has_surname:
        score += rules["scoring"]["surname_bonus"]

    return score


def choose_best_token_bundle(
    tokens: list[NameTokenCandidate],
    rules: RulesConfig,
) -> list[NameTokenCandidate]:
    """Выбирает лучший набор из 2-3 токенов по внутреннему скорингу."""
    if len(tokens) <= 3:
        return sorted(tokens, key=lambda item: item.line_idx)

    best_bundle: tuple[NameTokenCandidate, ...] | None = None
    best_score = float("-inf")

    for size in (3, 2):
        if len(tokens) < size:
            continue

        for bundle in itertools.combinations(tokens, size):
            bundle_score = score_token_bundle(bundle, rules)

            if bundle_score > best_score:
                best_score = bundle_score
                best_bundle = bundle

        if best_bundle is not None and len(best_bundle) == 3:
            break

    if best_bundle is None:
        return []

    return sorted(best_bundle, key=lambda item: item.line_idx)


def order_fio_tokens(
    tokens: list[NameTokenCandidate], rules: RulesConfig
) -> list[NameTokenCandidate]:
    """Упорядочивает токены в формате Фамилия Имя Отчество."""
    ordered = sorted(tokens, key=lambda item: item.line_idx)

    patronymics = [
        entry for entry in ordered if is_patronymic_token(entry.token, rules)
    ]
    non_patronymics = [entry for entry in ordered if entry not in patronymics]

    result: list[NameTokenCandidate] = []

    if non_patronymics:
        surname_candidates = [
            entry for entry in non_patronymics if is_surname_token(entry.token, rules)
        ]

        if surname_candidates and len(non_patronymics) >= 2:
            surname = min(surname_candidates, key=lambda item: item.line_idx)
            result.append(surname)

            rest = [entry for entry in non_patronymics if entry != surname]
            rest = sorted(rest, key=lambda item: item.line_idx)
            result.extend(rest)
        else:
            result.extend(sorted(non_patronymics, key=lambda item: item.line_idx))

    if patronymics:
        patronymic = max(
            patronymics, key=lambda item: (item.confidence, -item.line_idx)
        )

        result = [entry for entry in result if entry != patronymic]
        result.append(patronymic)

        for entry in sorted(
            [entry for entry in patronymics if entry != patronymic],
            key=lambda item: item.line_idx,
        ):
            result.append(entry)

    unique: list[NameTokenCandidate] = []

    for entry in result:
        if entry not in unique:
            unique.append(entry)

    return unique[:3]


def collect_anchor_candidates(
    lines: list[dict], rules: RulesConfig
) -> list[AnchorCandidate]:
    """Собирает кандидаты якоря владельца."""
    anchors: list[AnchorCandidate] = []
    line_meta: list[dict] = []

    for idx, line in enumerate(lines):
        score, has_owner, has_holder, normalized = get_owner_anchor_score(
            line["text"], rules
        )

        line_meta.append(
            {
                "idx": idx,
                "score": score,
                "has_owner": has_owner,
                "has_holder": has_holder,
                "normalized": normalized,
            }
        )

        if score >= rules["anchor"]["min_score"]:
            anchors.append(
                AnchorCandidate(
                    line_indices=(idx,),
                    center_idx=idx,
                    score=score,
                    source="single",
                )
            )

    max_gap = rules["anchor"]["split_max_gap"]

    for idx, current in enumerate(line_meta):
        if not (current["has_owner"] or current["has_holder"]):
            continue

        for offset in range(1, max_gap + 2):
            neighbour_idx = idx + offset

            if neighbour_idx >= len(line_meta):
                break

            neighbour = line_meta[neighbour_idx]

            if current["has_owner"] and neighbour["has_holder"]:
                score = (
                    max(current["score"], neighbour["score"])
                    + rules["anchor"]["split_anchor_bonus"]
                )
            elif current["has_holder"] and neighbour["has_owner"]:
                score = (
                    max(current["score"], neighbour["score"])
                    + rules["anchor"]["split_anchor_bonus"]
                )
            else:
                continue

            score = min(100.0, score)

            if score < rules["anchor"]["min_score"]:
                continue

            line_indices = tuple(sorted((idx, neighbour_idx)))
            center_idx = int(round(sum(line_indices) / len(line_indices)))

            anchors.append(
                AnchorCandidate(
                    line_indices=line_indices,
                    center_idx=center_idx,
                    score=score,
                    source="split",
                )
            )

    dedup: dict[tuple[int, ...], AnchorCandidate] = {}

    for anchor in anchors:
        existing = dedup.get(anchor.line_indices)

        if existing is None or anchor.score > existing.score:
            dedup[anchor.line_indices] = anchor

    sorted_anchors = sorted(dedup.values(), key=lambda item: item.score, reverse=True)

    return sorted_anchors[: rules["anchor"]["max_candidates"]]


def build_candidate_from_anchor(
    lines: list[dict],
    anchor: AnchorCandidate,
    rules: RulesConfig,
    rotation: int,
    ocr_variant: str,
) -> FioCandidate | None:
    """Строит кандидата ФИО вокруг одного якоря по окну соседних строк."""
    start_idx = max(0, min(anchor.line_indices) - rules["anchor"]["scan_window_before"])
    end_idx = min(
        len(lines) - 1, max(anchor.line_indices) + rules["anchor"]["scan_window_after"]
    )

    tokens: list[NameTokenCandidate] = []

    for idx in range(start_idx, end_idx + 1):
        if idx in anchor.line_indices:
            continue

        line = lines[idx]
        extracted_tokens = extract_name_tokens_from_line(line["text"], rules)

        if not extracted_tokens:
            continue

        distance = min(abs(idx - anchor_idx) for anchor_idx in anchor.line_indices)

        for token in extracted_tokens:
            add_unique_name_token(
                tokens,
                NameTokenCandidate(
                    token=token,
                    confidence=float(line["conf"]),
                    line_idx=idx,
                    distance=distance,
                    line_text=str(line["text"]),
                ),
            )

    if len(tokens) < 2:
        return None

    bundle = choose_best_token_bundle(tokens, rules)

    if len(bundle) < 2:
        return None

    ordered_bundle = order_fio_tokens(bundle, rules)

    if len(ordered_bundle) < 2:
        return None

    parts = [entry.token.title() for entry in ordered_bundle[:3]]
    confidences = [entry.confidence for entry in ordered_bundle[:3]]
    line_indices = [entry.line_idx for entry in ordered_bundle[:3]]
    distances = [entry.distance for entry in ordered_bundle[:3]]

    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    avg_distance = float(np.mean(distances)) if distances else 0.0
    line_span = max(line_indices) - min(line_indices) if line_indices else 0

    quality_score = len(parts) * rules["scoring"]["parts_weight"]
    quality_score += mean_confidence * rules["scoring"]["confidence_weight"]
    quality_score += anchor.score * rules["scoring"]["anchor_weight"]
    quality_score -= avg_distance * rules["scoring"]["distance_penalty"]
    quality_score -= line_span * rules["scoring"]["line_span_penalty"]

    if any(is_patronymic_token(entry.token, rules) for entry in ordered_bundle):
        quality_score += rules["scoring"]["patronymic_bonus"]

    if any(is_surname_token(entry.token, rules) for entry in ordered_bundle):
        quality_score += rules["scoring"]["surname_bonus"]

    for entry in ordered_bundle:
        if is_address_line(entry.line_text, rules):
            quality_score -= rules["scoring"]["address_penalty"]

        if is_service_line(entry.line_text, rules):
            quality_score -= rules["scoring"]["service_penalty"]

    return FioCandidate(
        surname=parts[0] if len(parts) > 0 else None,
        name=parts[1] if len(parts) > 1 else None,
        patronymic=parts[2] if len(parts) > 2 else None,
        confidence=mean_confidence,
        anchor_score=float(anchor.score),
        parts_count=len(parts),
        quality_score=float(quality_score),
        rotation=rotation,
        ocr_variant=ocr_variant,
    )


def extract_fio_from_lines(
    lines: list[dict],
    rules: RulesConfig,
    rotation: int,
    ocr_variant: str,
    debug_candidates: bool,
) -> dict | None:
    """Извлекает лучший кандидат ФИО из OCR-строк одного прохода."""
    if not lines:
        return None

    anchors = collect_anchor_candidates(lines, rules)

    if not anchors:
        if debug_candidates:
            logger.debug("Кандидаты якоря не найдены")
        return None

    candidates: list[FioCandidate] = []

    for anchor in anchors:
        candidate = build_candidate_from_anchor(
            lines, anchor, rules, rotation, ocr_variant
        )

        if candidate is None:
            continue

        if debug_candidates:
            logger.debug(
                "Кандидат ФИО: anchor_idx={}, source={}, score={:.2f}, fio='{} {} {}', quality={:.2f}",
                anchor.center_idx,
                anchor.source,
                anchor.score,
                candidate.surname,
                candidate.name,
                candidate.patronymic,
                candidate.quality_score,
            )

        candidates.append(candidate)

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item.quality_score)

    if best.parts_count < 2:
        return None

    if best.quality_score < rules["scoring"]["min_accept_score"]:
        return None

    return {
        "surname": best.surname,
        "name": best.name,
        "patronymic": best.patronymic,
        "confidence": best.confidence,
        "anchor_score": best.anchor_score,
        "parts_count": best.parts_count,
        "quality_score": best.quality_score,
        "rotation": best.rotation,
        "ocr_variant": best.ocr_variant,
    }


def run_ocr(img: np.ndarray, ocr_engine: PaddleOCR):
    """Запускает OCR и возвращает исходный результат движка."""
    start_time = time.perf_counter()
    result = ocr_engine.predict(img)
    elapsed = time.perf_counter() - start_time

    logger.info(
        "OCR завершен за {:.2f} сек. Результат пустой: {}",
        elapsed,
        not bool(result),
    )

    return result


def get_result_payload(page_result):
    """Унифицирует формат результата PaddleOCR для дальнейшего разбора."""
    if page_result is None:
        return None

    if isinstance(page_result, dict):
        return page_result.get("res", page_result)

    try:
        json_data = page_result.json
        if isinstance(json_data, dict):
            return json_data.get("res", json_data)
    except Exception:
        pass

    try:
        res_data = page_result["res"]
        if isinstance(res_data, dict):
            return res_data
    except Exception:
        pass

    try:
        payload = {
            "rec_texts": page_result["rec_texts"],
            "rec_scores": page_result["rec_scores"],
            "rec_polys": page_result["rec_polys"],
            "dt_polys": page_result["dt_polys"],
            "rec_boxes": page_result["rec_boxes"],
        }
        return payload
    except Exception:
        pass

    return None


def box_to_xy(box):
    """Преобразует bbox/полигон строки в списки X и Y координат."""
    arr = np.asarray(box)

    if arr.ndim == 1 and arr.shape[0] >= 4:
        x1, y1, x2, y2 = arr[:4]
        xs = [float(x1), float(x2)]
        ys = [float(y1), float(y2)]
        return xs, ys

    arr = arr.reshape(-1, 2)
    xs = arr[:, 0].astype(float).tolist()
    ys = arr[:, 1].astype(float).tolist()

    return xs, ys


def group_ocr_lines(ocr_result) -> list[dict]:
    """Преобразует OCR-ответ в отсортированный по Y список строк."""
    lines: list[dict] = []

    if not ocr_result:
        return lines

    for page_result in ocr_result:
        data = get_result_payload(page_result)

        if not data:
            continue

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        boxes = data.get("rec_polys")

        if boxes is None:
            boxes = data.get("dt_polys")

        if boxes is None:
            boxes = data.get("rec_boxes")

        if boxes is None:
            continue

        for text, conf, box in zip(texts, scores, boxes):
            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            xs, ys = box_to_xy(box)

            lines.append(
                {
                    "text": text,
                    "conf": float(conf),
                    "x1": min(xs),
                    "y1": min(ys),
                    "x2": max(xs),
                    "y2": max(ys),
                    "cy": (min(ys) + max(ys)) / 2,
                }
            )

    lines.sort(key=lambda item: item["cy"])
    return lines


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Поворачивает изображение на один из поддерживаемых углов."""
    if angle == 0:
        return img

    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)

    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unsupported angle: {angle}")


def preprocess(img: np.ndarray) -> np.ndarray:
    """Выполняет базовую предобработку изображения перед OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def extract_fio(
    image_path: Path,
    rules: RulesConfig,
    ocr_engine: PaddleOCR,
    debug_candidates: bool,
) -> dict | None:
    """Извлекает ФИО из одного изображения по всем углам и OCR-вариантам."""
    logger.info("{}", "=" * 80)
    logger.info("Начинаем обработку изображения: {}", image_path)

    image = cv2.imread(str(image_path), 1)

    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    best: dict | None = None
    best_score = float("-inf")

    for angle in rules["ocr"]["angles"]:
        rotated = rotate_image(image, angle)

        variant_cache: dict[str, np.ndarray] = {"raw": rotated}

        for variant in rules["ocr"]["variants"]:
            if variant not in {"raw", "preprocessed"}:
                logger.warning("Неизвестный OCR-вариант в config.py: {}", variant)
                continue

            if variant == "preprocessed" and variant not in variant_cache:
                variant_cache[variant] = preprocess(rotated)

            image_variant = variant_cache[variant]

            logger.info("OCR-вариант: angle={}, variant={}", angle, variant)

            ocr_result = run_ocr(image_variant, ocr_engine)

            if not ocr_result:
                continue

            lines = group_ocr_lines(ocr_result)

            if not lines:
                continue

            candidate = extract_fio_from_lines(
                lines=lines,
                rules=rules,
                rotation=angle,
                ocr_variant=variant,
                debug_candidates=debug_candidates,
            )

            if not candidate:
                continue

            quality_score = float(candidate.get("quality_score", 0.0))

            logger.info(
                "Кандидат найден: angle={}, variant={}, fio='{} {} {}', quality={:.2f}",
                angle,
                variant,
                candidate.get("surname"),
                candidate.get("name"),
                candidate.get("patronymic"),
                quality_score,
            )

            if quality_score > best_score:
                best_score = quality_score
                best = candidate

    if best:
        logger.success(
            "Итог по файлу: OK. file={}, fio='{} {} {}', confidence={:.4f}, rotation={}, variant={}",
            image_path,
            best.get("surname"),
            best.get("name"),
            best.get("patronymic"),
            best.get("confidence", 0.0),
            best.get("rotation"),
            best.get("ocr_variant"),
        )
    else:
        logger.warning("Итог по файлу: ФИО не найдено. file={}", image_path)

    return best


def iter_image_files(
    input_folder: str | Path, rules: RulesConfig, recursive: bool = False
) -> list[Path]:
    """Возвращает список файлов изображений из входной папки."""
    folder = Path(input_folder)

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {folder}")

    pattern = "**/*" if recursive else "*"

    image_files = [
        path
        for path in folder.glob(pattern)
        if path.is_file()
        and path.suffix.lower() in set(rules["io"]["image_extensions"])
    ]

    return sorted(image_files)


def build_csv_row(
    image_path: Path, result: dict | None, status: str, error: str = ""
) -> dict:
    """Формирует строку для итогового CSV."""
    if result:
        surname = result.get("surname") or ""
        name = result.get("name") or ""
        patronymic = result.get("patronymic") or ""
        fio = " ".join(part for part in [surname, name, patronymic] if part)

        return {
            "file": str(image_path),
            "surname": surname,
            "name": name,
            "patronymic": patronymic,
            "fio": fio,
            "confidence": result.get("confidence", ""),
            "rotation": result.get("rotation", ""),
            "status": status,
            "error": error,
        }

    return {
        "file": str(image_path),
        "surname": "",
        "name": "",
        "patronymic": "",
        "fio": "",
        "confidence": "",
        "rotation": "",
        "status": status,
        "error": error,
    }


def process_folder(
    input_folder: str | Path,
    output_csv: str | Path,
    rules: RulesConfig,
    recursive: bool,
    debug_candidates: bool,
) -> int:
    """Обрабатывает папку изображений и сохраняет результаты в CSV."""
    start_time = time.perf_counter()

    ocr_engine = get_ocr_engine()
    image_files = iter_image_files(input_folder, rules, recursive=recursive)

    fieldnames = [
        "file",
        "surname",
        "name",
        "patronymic",
        "fio",
        "confidence",
        "rotation",
        "status",
        "error",
    ]

    rows: list[dict] = []

    ok_count = 0
    not_found_count = 0
    error_count = 0

    for image_path in image_files:
        try:
            result = extract_fio(
                image_path=image_path,
                rules=rules,
                ocr_engine=ocr_engine,
                debug_candidates=debug_candidates,
            )

            if result:
                rows.append(build_csv_row(image_path, result, status="OK"))
                ok_count += 1
            else:
                rows.append(
                    build_csv_row(
                        image_path, None, status="NOT_FOUND", error="ФИО не найдено"
                    )
                )
                not_found_count += 1

        except Exception as exc:
            logger.exception("Ошибка при обработке файла: {}", image_path)
            rows.append(build_csv_row(image_path, None, status="ERROR", error=str(exc)))
            error_count += 1

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=rules["io"]["csv_delimiter"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - start_time

    logger.success("CSV сохранен: {}", output_path)
    logger.info("Всего файлов: {}", len(image_files))
    logger.info("OK: {}", ok_count)
    logger.info("NOT_FOUND: {}", not_found_count)
    logger.info("ERROR: {}", error_count)
    logger.info("Общее время обработки: {:.2f} сек.", elapsed)

    return len(rows)
