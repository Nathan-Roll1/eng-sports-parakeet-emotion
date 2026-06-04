from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import imageio_ffmpeg

    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"


@dataclass
class StationInfo:
    stationuuid: str
    name: str
    url: str
    url_resolved: str
    country: str
    countrycode: str
    language: str
    language_canonical: str
    tags: str
    codec: str
    bitrate: int
    votes: int
    homepage: str


@dataclass
class RecordingConfig:
    duration_seconds: int = 1800
    sample_rate: int = 16000
    channels: int = 1
    output_format: str = "flac"
    max_concurrent: int = 50
    output_dir: str = "recordings"
    checkpoint_file: str = "checkpoint.json"
    max_retries: int = 3
    retry_delay: float = 5.0
    ffmpeg_timeout: int = 1860
    bandwidth_test_seconds: int = 10
    min_audio_bytes: int = 10000
    min_duration: float = 1700.0
    max_silence_ratio: float = 0.5


@dataclass
class RecordingResult:
    station: StationInfo
    filepath: str | None
    duration_actual: float
    file_size_bytes: int
    started_at: str
    completed_at: str
    success: bool
    error: str | None
    avg_bitrate_kbps: float
    silence_ratio: float
    peak_amplitude: float


@dataclass
class BandwidthProfile:
    tested_at: str
    download_speed_mbps: float
    recommended_concurrent: int
    estimated_total_hours: float
    total_stations: int


LANGUAGE_NORMALIZATION: dict[str, str] = {
    "english": "English",
    "engilsh": "English",
    "englsh": "English",
    "englisch": "English",
    "british english": "English",
    "american english": "English",
    "en": "English",
    "eng": "English",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "castellano": "Spanish",
    "spanis": "Spanish",
    "spanisch": "Spanish",
    "es": "Spanish",
    "french": "French",
    "français": "French",
    "francais": "French",
    "francaise": "French",
    "française": "French",
    "fr": "French",
    "german": "German",
    "deutsch": "German",
    "deutch": "German",
    "germen": "German",
    "de": "German",
    "italian": "Italian",
    "italiano": "Italian",
    "itallian": "Italian",
    "it": "Italian",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "portugues": "Portuguese",
    "portugese": "Portuguese",
    "pt": "Portuguese",
    "dutch": "Dutch",
    "nederlands": "Dutch",
    "holland": "Dutch",
    "nederlandse": "Dutch",
    "nl": "Dutch",
    "russian": "Russian",
    "русский": "Russian",
    "rusian": "Russian",
    "ru": "Russian",
    "polish": "Polish",
    "polski": "Polish",
    "polsk": "Polish",
    "pl": "Polish",
    "ukrainian": "Ukrainian",
    "українська": "Ukrainian",
    "ukranian": "Ukrainian",
    "ua": "Ukrainian",
    "czech": "Czech",
    "čeština": "Czech",
    "cestina": "Czech",
    "czeck": "Czech",
    "cs": "Czech",
    "slovak": "Slovak",
    "slovenčina": "Slovak",
    "slovencina": "Slovak",
    "slovac": "Slovak",
    "sk": "Slovak",
    "hungarian": "Hungarian",
    "magyar": "Hungarian",
    "hungaran": "Hungarian",
    "hu": "Hungarian",
    "romanian": "Romanian",
    "română": "Romanian",
    "romana": "Romanian",
    "rumanian": "Romanian",
    "ro": "Romanian",
    "greek": "Greek",
    "ελληνικά": "Greek",
    "ellinika": "Greek",
    "greak": "Greek",
    "el": "Greek",
    "turkish": "Turkish",
    "türkçe": "Turkish",
    "turkce": "Turkish",
    "turkisch": "Turkish",
    "tr": "Turkish",
    "arabic": "Arabic",
    "العربية": "Arabic",
    "arab": "Arabic",
    "ar": "Arabic",
    "hebrew": "Hebrew",
    "עברית": "Hebrew",
    "ivrit": "Hebrew",
    "he": "Hebrew",
    "hindi": "Hindi",
    "हिन्दी": "Hindi",
    "hindhi": "Hindi",
    "hi": "Hindi",
    "bengali": "Bengali",
    "বাংলা": "Bengali",
    "bangla": "Bengali",
    "bn": "Bengali",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "cantonese": "Chinese",
    "中文": "Chinese",
    "chineese": "Chinese",
    "zh": "Chinese",
    "japanese": "Japanese",
    "日本語": "Japanese",
    "nihongo": "Japanese",
    "japaneese": "Japanese",
    "ja": "Japanese",
    "korean": "Korean",
    "한국어": "Korean",
    "hangug-eo": "Korean",
    "ko": "Korean",
    "thai": "Thai",
    "ไทย": "Thai",
    "thailan": "Thai",
    "th": "Thai",
    "vietnamese": "Vietnamese",
    "tiếng việt": "Vietnamese",
    "tieng viet": "Vietnamese",
    "vietnameese": "Vietnamese",
    "vi": "Vietnamese",
    "indonesian": "Indonesian",
    "bahasa": "Indonesian",
    "bahasa indonesia": "Indonesian",
    "indonesion": "Indonesian",
    "id": "Indonesian",
    "malay": "Malay",
    "bahasa melayu": "Malay",
    "melayu": "Malay",
    "ms": "Malay",
    "swedish": "Swedish",
    "svenska": "Swedish",
    "sweedish": "Swedish",
    "sv": "Swedish",
    "norwegian": "Norwegian",
    "norsk": "Norwegian",
    "norvegian": "Norwegian",
    "no": "Norwegian",
    "danish": "Danish",
    "dansk": "Danish",
    "danisch": "Danish",
    "da": "Danish",
    "finnish": "Finnish",
    "suomi": "Finnish",
    "finnishh": "Finnish",
    "fi": "Finnish",
    "icelandic": "Icelandic",
    "íslenska": "Icelandic",
    "islenska": "Icelandic",
    "icelandc": "Icelandic",
    "is": "Icelandic",
    "croatian": "Croatian",
    "hrvatski": "Croatian",
    "croatain": "Croatian",
    "hr": "Croatian",
    "serbian": "Serbian",
    "српски": "Serbian",
    "srpski": "Serbian",
    "sr": "Serbian",
    "bulgarian": "Bulgarian",
    "български": "Bulgarian",
    "bulgaran": "Bulgarian",
    "bg": "Bulgarian",
    "slovenian": "Slovenian",
    "slovenščina": "Slovenian",
    "slovenscina": "Slovenian",
    "slovanian": "Slovenian",
    "sl": "Slovenian",
    "lithuanian": "Lithuanian",
    "lietuvių": "Lithuanian",
    "lietuviu": "Lithuanian",
    "lithuanina": "Lithuanian",
    "lt": "Lithuanian",
    "latvian": "Latvian",
    "latviešu": "Latvian",
    "latviesu": "Latvian",
    "lativan": "Latvian",
    "lv": "Latvian",
    "estonian": "Estonian",
    "eesti": "Estonian",
    "estoniann": "Estonian",
    "et": "Estonian",
    "irish": "Irish",
    "gaeilge": "Irish",
    "irland": "Irish",
    "ga": "Irish",
    "welsh": "Welsh",
    "cymraeg": "Welsh",
    "cy": "Welsh",
    "catalan": "Catalan",
    "català": "Catalan",
    "catala": "Catalan",
    "ca": "Catalan",
    "basque": "Basque",
    "euskera": "Basque",
    "euskara": "Basque",
    "eu": "Basque",
    "persian": "Persian",
    "farsi": "Persian",
    "فارسی": "Persian",
    "fa": "Persian",
    "urdu": "Urdu",
    "اردو": "Urdu",
    "ur": "Urdu",
    "tamil": "Tamil",
    "தமிழ்": "Tamil",
    "ta": "Tamil",
    "telugu": "Telugu",
    "తెలుగు": "Telugu",
    "te": "Telugu",
    "punjabi": "Punjabi",
    "ਪੰਜਾਬੀ": "Punjabi",
    "panjabi": "Punjabi",
    "pa": "Punjabi",
    "swahili": "Swahili",
    "kiswahili": "Swahili",
    "swaheli": "Swahili",
    "sw": "Swahili",
    "albanian": "Albanian",
    "shqip": "Albanian",
    "albanain": "Albanian",
    "sq": "Albanian",
    "macedonian": "Macedonian",
    "македонски": "Macedonian",
    "macedoinan": "Macedonian",
    "mk": "Macedonian",
    "afrikaans": "Afrikaans",
    "afrikaan": "Afrikaans",
    "af": "Afrikaans",
}


def normalize_language(raw: str) -> str:
    s = raw.strip()
    if not s:
        return s
    for part in s.split(","):
        key = " ".join(part.strip().lower().split())
        if not key:
            continue
        canonical = LANGUAGE_NORMALIZATION.get(key)
        if canonical is not None:
            return canonical
    return s.title()


def get_output_path(station: StationInfo, config: RecordingConfig) -> Path:
    ext = config.output_format.removeprefix(".").lower() or "flac"
    return (
        Path(config.output_dir)
        / station.countrycode
        / station.language_canonical
        / f"{station.stationuuid}.{ext}"
    )
