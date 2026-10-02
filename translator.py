"""번역 로직. OpenAI API 호출과 응답(JSON) 파싱을 담당한다."""

import json
import os

import openai
from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-6-astra"
# 5,000자 번역에 약 100초가 걸리므로 여유를 두되, 응답이 없을 때 무한정 기다리지 않도록 제한한다
REQUEST_TIMEOUT_SECONDS = 180

# 지원 언어: 코드 -> 표시 이름 (언어 목록은 여기서만 관리)
LANGUAGES = {
    "en": "영어",
    "ja": "일본어",
    "vi": "베트남어",
}

# 프롬프트에 넣을 언어의 영어 이름
_LANGUAGE_NAMES_EN = {
    "en": "English",
    "ja": "Japanese",
    "vi": "Vietnamese",
}

SYSTEM_PROMPT = """You are a professional translator.
- Automatically detect the source language of the user's text.
- Translate it into each requested target language.
- Preserve the original meaning, tone, and line breaks exactly.
- Keep numbers, URLs, and code as they are.
- Write names of people, places, and organizations in the target language's usual form: romanize names written in a non-Latin script for English and Vietnamese (e.g. 김민수 -> Kim Min-su), and use katakana or the customary kanji for Japanese.
- If the text is already in a target language, return it unchanged for that language.
- Respond with a single JSON object only, no explanations. Its keys are the requested language codes and its values are the translations."""

load_dotenv()


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 메시지를 담은 번역 오류."""


def _get_setting(name: str) -> str | None:
    """환경변수(.env)에서 먼저 찾고, 없으면 st.secrets에서 찾는다."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다
        return None


def get_api_key() -> str | None:
    return _get_setting("OPENAI_API_KEY")


def get_model() -> str:
    return _get_setting("OPENAI_MODEL") or DEFAULT_MODEL


def _build_user_prompt(text: str, target_langs: list[str]) -> str:
    targets = ", ".join(f'"{code}" ({_LANGUAGE_NAMES_EN[code]})' for code in target_langs)
    return f"Target languages: {targets}\n\nText to translate:\n{text}"


def _parse_response(content: str | None, target_langs: list[str]) -> dict[str, str]:
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as e:
        raise TranslationError("번역 결과를 해석하지 못했습니다. 다시 시도해 주세요.") from e
    if not isinstance(data, dict):
        raise TranslationError("번역 결과 형식이 올바르지 않습니다. 다시 시도해 주세요.")

    missing = [code for code in target_langs if not isinstance(data.get(code), str)]
    if missing:
        names = ", ".join(LANGUAGES[code] for code in missing)
        raise TranslationError(f"일부 언어({names})의 번역 결과가 없습니다. 다시 시도해 주세요.")
    return {code: data[code] for code in target_langs}


def translate(text: str, target_langs: list[str]) -> dict[str, str]:
    """text를 target_langs(언어 코드 목록)로 번역해 {코드: 번역문}을 반환한다.

    선택된 모든 언어를 한 번의 API 호출로 번역한다. 실패하면 TranslationError를 던진다.
    """
    unknown = [code for code in target_langs if code not in LANGUAGES]
    if unknown:
        raise ValueError(f"지원하지 않는 언어 코드: {unknown}")

    api_key = get_api_key()
    if not api_key:
        raise TranslationError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 또는 Secrets를 확인해 주세요.")

    client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=1)
    try:
        response = client.chat.completions.create(
            model=get_model(),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(text, target_langs)},
            ],
        )
    except openai.AuthenticationError as e:
        raise TranslationError("API 키 인증에 실패했습니다. OPENAI_API_KEY 값을 확인해 주세요.") from e
    except openai.RateLimitError as e:
        raise TranslationError("요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.") from e
    except openai.NotFoundError as e:
        raise TranslationError(f"모델 '{get_model()}'을(를) 찾을 수 없습니다. OPENAI_MODEL 값을 확인해 주세요.") from e
    except openai.BadRequestError as e:
        raise TranslationError("번역 요청이 거부되었습니다. 입력 내용이나 모델 설정을 확인해 주세요.") from e
    except openai.APITimeoutError as e:
        raise TranslationError("번역 응답이 너무 오래 걸려 중단했습니다. 텍스트를 나누어 다시 시도해 주세요.") from e
    except openai.APIConnectionError as e:
        raise TranslationError("OpenAI 서버에 연결하지 못했습니다. 네트워크 상태를 확인해 주세요.") from e
    except openai.APIError as e:
        raise TranslationError("번역 중 API 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.") from e

    return _parse_response(response.choices[0].message.content, target_langs)
