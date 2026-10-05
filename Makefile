# Makefile

.PHONY: all format check validate test ty bandit fix reformat fix-ruff

# Tests that run without API keys or a browser.
OFFLINE_TESTS := \
	tests/test_spatial_drag_helpers.py \
	tests/test_prompt_contracts.py \
	tests/test_schema_image_binary_challenge.py \
	tests/test_refine_click_point.py \
	tests/test_request_debug_cache.py \
	tests/test_helper_create_coordinate_grid.py \
	tests/test_helper_env_generator.py \
	tests/test_helper_visualize_attention_points.py \
	tests/test_helper_create_comparison_image.py \
	tests/test_helper_webm_to_mp4.py

# Default target: runs format and check
all: validate test

format:
	uv run black --check src/ tests/

reformat:
	uv run black src/ tests/

check:
	uv run ruff check .

fix-ruff:
	uv run ruff check . --fix

fix: reformat fix-ruff

test:
	uv run pytest $(OFFLINE_TESTS) -q

ty:
	uv run ty check src/

bandit:
	uv run bandit -r src/ -lll -q

validate: format check ty bandit
	@echo "Validation passed. Your code is ready to push."
