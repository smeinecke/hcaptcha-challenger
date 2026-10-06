import unicodedata

from pydantic import BaseModel, Field

from hcaptcha_challenger.models import JobTypeLiteral

# Cyrillic/Greek/fullwidth lookalikes captcha prompts substitute for ASCII.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "с": "c",
        "е": "e",
        "і": "i",
        "ј": "j",
        "о": "o",
        "р": "p",
        "ѕ": "s",
        "х": "x",
        "у": "y",
        "в": "b",
        "к": "k",
        "м": "m",
        "н": "h",
        "т": "t",
        "һ": "h",
        "ԁ": "d",
        "ɡ": "g",
        "ԛ": "q",
        "ԝ": "w",
        "ο": "o",
        "α": "a",
        "ε": "e",
        "ι": "i",
        "ν": "v",
        "ρ": "p",
        "τ": "t",
        "χ": "x",
        "υ": "u",
        "ω": "w",
        "ｏ": "o",
        "０": "0",
    }
)


def _normalize_text(text: str) -> str:
    """NFKD-normalize, map common homoglyphs, strip combining marks, lowercase."""
    text = text.translate(_HOMOGLYPHS)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


class SkillRule(BaseModel):
    """Represents a single skill matching rule."""

    triggers: list[str] = Field(...)
    job_type: JobTypeLiteral | None = Field(default=None)
    template: str = Field(...)

    # Pre-computed lowercase triggers for faster matching
    _triggers_lower: list[str] | None = None

    def model_post_init(self, __context, /) -> None:
        """Pre-compute normalized triggers after model initialization."""
        object.__setattr__(self, "_triggers_lower", [_normalize_text(t) for t in self.triggers])

    def matches_text(self, text_lower: str) -> bool:
        """Check if all triggers match the given text (AND logic, homoglyph-safe)."""
        text_norm = _normalize_text(text_lower)
        triggers = self._triggers_lower or [_normalize_text(t) for t in self.triggers]
        return all(trigger in text_norm for trigger in triggers)


class SkillManifest(BaseModel):
    """Represents the skill manifest containing version and rules."""

    version: str = Field(...)
    base_url: str | None = Field(default=None)
    rules: list[SkillRule] = Field(...)

    @staticmethod
    def get_download_url(repo: str, branch: str = "main") -> str:
        """Construct the raw GitHub URL for this manifest."""
        return f"https://raw.githubusercontent.com/{repo}/{branch}/src/hcaptcha_challenger/skills/rules.yaml"

    def get_library_base_url(self, repo: str, branch: str = "main") -> str:
        """Get the base URL for downloading template files."""
        return (
            self.base_url
            or f"https://raw.githubusercontent.com/{repo}/{branch}/src/hcaptcha_challenger/skills/library"
        )
