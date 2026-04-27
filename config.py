from copy import deepcopy
from typing import Any, TypeAlias


RulesConfig: TypeAlias = dict[str, Any]


LATIN_TO_CYR_MAP = {
    "A": "А",
    "B": "В",
    "C": "С",
    "D": "Д",
    "E": "Е",
    "H": "Н",
    "I": "И",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "U": "У",
    "V": "В",
    "X": "Х",
    "Y": "У",
}

FIELD_WORDS = frozenset(
    {
        "РЕСПУБЛИКА",
        "КРАЙ",
        "ОБЛАСТЬ",
        "СУБЪЕКТ",
        "РОССИЙСКОЙ",
        "ФЕДЕРАЦИИ",
        "МОСКВА",
        "РАЙОН",
        "НАС",
        "ПУНКТ",
        "НАСЕЛЕННЫЙ",
        "УЛИЦА",
        "ДОМ",
        "КОРП",
        "КВ",
        "КВАРТИРА",
        "ОСОБЫЕ",
        "ОТМЕТКИ",
        "КОД",
        "ПОДРАЗДЕЛЕНИЯ",
        "ГИБДД",
        "ДАТА",
        "ВЫДАЧИ",
        "ПОДПИСЬ",
        "ВЫДАНО",
    }
)

ADDRESS_STOP_WORDS = frozenset(
    {
        "РЕСПУБЛИКА",
        "СУБЪЕКТ",
        "МОСКВА",
        "УЛИЦА",
        "ДОМ",
        "ОСОБЫЕ",
        "ОБЛАСТЬ",
        "РАЙОН",
        "КРАЙ",
        "КВАРТИРА",
        "КОРП",
        "КОД",
        "ПОДРАЗДЕЛЕНИЯ",
        "НАС",
        "ПУНКТ",
    }
)

SERVICE_STOP_WORDS = frozenset(
    {
        "ГИБДД",
        "КОД",
        "ПОДРАЗДЕЛЕНИЯ",
        "ВЫДАНО",
        "ПОДПИСЬ",
        "ОТМЕТКИ",
        "КВ",
        "СТС",
    }
)

HARD_REJECT_SUBSTRINGS = (
    "ГИБДД",
    "ПОДРАЗДЕЛЕНИ",
    "РЕСПУ",
    "ОБЛАСТ",
    "УЛИЦ",
    "ДОМ",
    "КВ",
    "КОРП",
    "КОД",
    "МОСКВ",
    "РАЙОН",
    "ОТМЕТК",
    "ВЫДАЧ",
    "ПОДПИС",
    "СТС",
    "ДОГОВОР",
    "ДАТА",
)

PATRONYMIC_SUFFIXES = (
    "ОВИЧ",
    "ЕВИЧ",
    "ИЧ",
    "ОВНА",
    "ЕВНА",
    "ИЧНА",
    "ЫЧ",
    "КЫЗЫ",
    "ОГЛЫ",
)

SURNAME_SUFFIXES = (
    "ОВ",
    "ЕВ",
    "ИН",
    "ЫН",
    "СКИЙ",
    "СКАЯ",
    "ОВА",
    "ЕВА",
    "ИНА",
    "ЫНА",
    "УК",
    "ЮК",
    "КО",
    "ЕНКО",
    "ШВИЛИ",
    "ДЗЕ",
    "ЯН",
    "АВА",
)

SUFFIX_CORRECTIONS = {
    "ОБНА": "ОВНА",
    "ЕБНА": "ЕВНА",
    "ОБИЧ": "ОВИЧ",
    "ЕБИЧ": "ЕВИЧ",
}

RULES: RulesConfig = {
    "io": {
        "image_extensions": tuple(
            sorted({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
        ),
        "csv_delimiter": ";",
    },
    "ocr": {
        "angles": (0, 90, 180, 270),
        "variants": ("raw", "preprocessed"),
    },
    "text": {
        "min_name_token_len": 2,
        "max_name_token_len": 35,
        "max_tokens_per_line": 4,
        "max_unconverted_latin_ratio": 0.34,
        "latin_to_cyr_table": str.maketrans(LATIN_TO_CYR_MAP),
    },
    "anchor": {
        "phrases": (
            "СОБСТВЕННИК",
            "СОБСТВЕННИК ВЛАДЕЛЕЦ",
            "СОБСТВЕННИК (ВЛАДЕЛЕЦ)",
            "ВЛАДЕЛЕЦ",
        ),
        "min_score": 73.0,
        "min_cyr_letters": 6,
        "weaker_keyword_penalty": 25.0,
        "owner_keyword_bonus": 8.0,
        "split_anchor_bonus": 6.0,
        "split_max_gap": 1,
        "scan_window_before": 18,
        "scan_window_after": 18,
        "max_candidates": 10,
    },
    "filters": {
        "field_words": FIELD_WORDS,
        "address_stop_words": ADDRESS_STOP_WORDS,
        "service_stop_words": SERVICE_STOP_WORDS,
        "hard_reject_substrings": HARD_REJECT_SUBSTRINGS,
    },
    "postprocess": {
        "patronymic_suffixes": PATRONYMIC_SUFFIXES,
        "surname_suffixes": SURNAME_SUFFIXES,
        "suffix_corrections": SUFFIX_CORRECTIONS,
    },
    "scoring": {
        "parts_weight": 100.0,
        "confidence_weight": 45.0,
        "anchor_weight": 0.24,
        "patronymic_bonus": 12.0,
        "surname_bonus": 8.0,
        "distance_penalty": 1.8,
        "line_span_penalty": 1.2,
        "address_penalty": 25.0,
        "service_penalty": 20.0,
        "min_accept_score": 170.0,
    },
}

RULES["all_stop_words"] = (
    RULES["filters"]["field_words"]
    | RULES["filters"]["address_stop_words"]
    | RULES["filters"]["service_stop_words"]
)


def _validate_rules(rules: RulesConfig) -> None:
    if not rules["io"]["image_extensions"]:
        raise ValueError("io.image_extensions не должен быть пустым")

    if not rules["io"]["csv_delimiter"]:
        raise ValueError("io.csv_delimiter не должен быть пустым")

    if rules["text"]["min_name_token_len"] < 1:
        raise ValueError("text.min_name_token_len должен быть >= 1")

    if rules["text"]["max_name_token_len"] < rules["text"]["min_name_token_len"]:
        raise ValueError(
            "text.max_name_token_len должен быть >= text.min_name_token_len"
        )

    ratio = rules["text"]["max_unconverted_latin_ratio"]
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(
            "text.max_unconverted_latin_ratio должен быть в диапазоне [0, 1]"
        )

    if not rules["anchor"]["phrases"]:
        raise ValueError("anchor.phrases не должен быть пустым")

    if rules["anchor"]["min_score"] <= 0:
        raise ValueError("anchor.min_score должен быть > 0")

    if rules["anchor"]["max_candidates"] < 1:
        raise ValueError("anchor.max_candidates должен быть >= 1")

    if any(angle not in {0, 90, 180, 270} for angle in rules["ocr"]["angles"]):
        raise ValueError("ocr.angles может содержать только 0, 90, 180, 270")

    if not rules["ocr"]["variants"]:
        raise ValueError("ocr.variants не должен быть пустым")

    if not rules["postprocess"]["patronymic_suffixes"]:
        raise ValueError("postprocess.patronymic_suffixes не должен быть пустым")

    if not (
        rules["filters"]["field_words"]
        or rules["filters"]["address_stop_words"]
        or rules["filters"]["service_stop_words"]
    ):
        raise ValueError("Должен быть задан хотя бы один набор стоп-слов")

    if rules["scoring"]["min_accept_score"] <= 0:
        raise ValueError("scoring.min_accept_score должен быть > 0")


_validate_rules(RULES)


def load_rules(_: str | None = None) -> RulesConfig:
    return deepcopy(RULES)
