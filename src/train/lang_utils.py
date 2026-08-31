"""Minimal language helpers for inference (extracted from dataset_hf)."""

LANG_CODE_TO_NAME = {
    # Chinese
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "chinese": "Chinese", "cn": "Chinese", "zhs": "Chinese", "zht": "Chinese",
    # English
    "en": "English", "en-us": "English", "en-gb": "English",
    "english": "English", "eng": "English",
    # German
    "de": "German", "de-de": "German", "german": "German", "deu": "German",
    # Spanish
    "es": "Spanish", "es-es": "Spanish", "spanish": "Spanish", "spa": "Spanish",
    # Japanese
    "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese", "jpn": "Japanese",
    # French
    "fr": "French", "fr-fr": "French", "french": "French", "fra": "French",
    # Italian
    "it": "Italian", "it-it": "Italian", "italian": "Italian", "ita": "Italian",
    # Portuguese
    "pt": "Portuguese", "pt-br": "Portuguese", "portuguese": "Portuguese", "por": "Portuguese",
    # Russian
    "ru": "Russian", "ru-ru": "Russian", "russian": "Russian", "rus": "Russian",
    # Korean
    "ko": "Korean", "kr": "Korean", "korean": "Korean", "kor": "Korean",
}


def normalize_lang(lang: str) -> str:
    """
    Normalize language code to full name.
    Examples:
        "zh" -> "Chinese"
        "DE" -> "German"
        "English" -> "English"
    """
    lang_lower = lang.lower().strip()
    if lang_lower in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[lang_lower]
    # If already a full name (first letter uppercase), return as-is
    if lang and lang[0].isupper() and lang_lower not in LANG_CODE_TO_NAME:
        return lang
    logger.warning(f"Unknown language code '{lang}', using as-is")
    return lang
