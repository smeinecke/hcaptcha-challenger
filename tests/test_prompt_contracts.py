from pathlib import Path


ACTIVE_PROMPTS = [
    Path("src/hcaptcha_challenger/tools/challenge_router/challenge_router.md"),
    Path("src/hcaptcha_challenger/tools/image_classifier/image_classifier.md"),
    Path("src/hcaptcha_challenger/tools/spatial/point.md"),
    Path("src/hcaptcha_challenger/tools/spatial/bbox.md"),
    Path("src/hcaptcha_challenger/tools/spatial/path.md"),
    Path("src/hcaptcha_challenger/skills/library/drag_connection.md"),
    Path("src/hcaptcha_challenger/skills/library/drag_object_to_shadow_v3_cn.md"),
    Path("src/hcaptcha_challenger/skills/library/drag_fragment_fit.md"),
    Path("src/hcaptcha_challenger/skills/library/drag_pairs.md"),
    Path("src/hcaptcha_challenger/skills/library/drag_similar.md"),
    Path("src/hcaptcha_challenger/skills/library/label_holes.md"),
]


def test_active_prompts_are_non_empty():
    for path in ACTIVE_PROMPTS:
        text = path.read_text(encoding="utf-8").strip()
        assert text, f"{path} should not be empty"


def test_router_prompt_keeps_all_enum_labels():
    text = Path(
        "src/hcaptcha_challenger/tools/challenge_router/challenge_router.md"
    ).read_text(encoding="utf-8")
    for label in [
        "image_label_single_select",
        "image_label_multi_select",
        "image_drag_single",
        "image_drag_multi",
    ]:
        assert label in text


def test_prompt_files_keep_required_schema_keywords():
    expectations = {
        "src/hcaptcha_challenger/tools/challenge_router/challenge_router.md": [
            "challenge_prompt",
            "challenge_type",
        ],
        "src/hcaptcha_challenger/tools/image_classifier/image_classifier.md": [
            "ImageBinaryChallenge",
            "challenge_prompt",
            "coordinates",
            "box_2d",
        ],
        "src/hcaptcha_challenger/tools/spatial/point.md": ["ImageAreaSelectChallenge"],
        "src/hcaptcha_challenger/tools/spatial/bbox.md": [
            "ImageBboxChallenge",
            "challenge_prompt",
        ],
        "src/hcaptcha_challenger/tools/spatial/path.md": ["ImageDragDropChallenge"],
    }

    for filename, keywords in expectations.items():
        text = Path(filename).read_text(encoding="utf-8")
        for keyword in keywords:
            assert keyword in text, f"{filename} should contain {keyword}"
