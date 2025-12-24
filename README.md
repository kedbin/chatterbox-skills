# Chatterbox OpenCode Skills

This repository contains custom agent skills for OpenCode to automate the generation of voiceover scripts and audio using Chatterbox TTS (Turbo).

## Skills Included

### 1. Create Script (`.opencode/skill/create-script`)
Transforms raw content (URLs, text, notes) into a natural, conversational script optimized for voiceover.
- Preserves acronyms (e.g., AI).
- Cleans up structure for better flow.
- Adds paralinguistic tags like `[chuckle]` or `[laugh]`.

### 2. Voiceover (`.opencode/skill/voiceover`)
Generates audio from a script using voice cloning via Chatterbox Turbo.
- Supports paralinguistic tags.
- Efficiently processes long texts in chunks.
- Uses `uv run` for consistent environment execution.

## Quickstart

1. **Prerequisites**:
   - Install [uv](https://github.com/astral-sh/uv).
   - Ensure `chatterbox` and its dependencies are installed in your environment.
   - A voice sample file named `clone.wav` in your root directory.

2. **Installation**:
   Copy the `.opencode` directory to your project root.

3. **Usage**:
   Simply ask your OpenCode agent:
   > "Using your skills, generate me a script from this article [URL] and do a voice clone on it."

## Repository Structure
- `.opencode/skill/`: Contains the skill definitions.
- `scripts/voiceover_script.py`: The underlying Python script for audio generation.
- `examples/entry-009.wav`: Sample output generated from a journal entry.

## License
Apache-2.0
