import torch
import scipy.io.wavfile as wavfile
from chatterbox.tts_turbo import ChatterboxTurboTTS
import os
import re
import pyloudnorm as ln
import numpy as np
import math
import argparse


def norm_loudness(wav, sr, target_lufs=-19):
    """
    Normalize audio to target LUFS using pyloudnorm.
    """
    try:
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(wav):
            wav_np = wav.cpu().numpy()
            if wav_np.ndim == 2 and wav_np.shape[0] == 1:
                wav_np = wav_np.squeeze(0)
        else:
            wav_np = wav

        meter = ln.Meter(sr)
        loudness = meter.integrated_loudness(wav_np)
        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        if math.isfinite(gain_linear) and gain_linear > 0.0:
            wav_np = wav_np * gain_linear

        if torch.is_tensor(wav):
            return (
                torch.from_numpy(wav_np).unsqueeze(0)
                if wav.ndim == 2
                else torch.from_numpy(wav_np)
            )
        return wav_np
    except Exception as e:
        print(f"Warning: Error in norm_loudness, skipping: {e}")
        return wav


def chunk_text(text, max_chars=250):
    """
    Split text into chunks based on punctuation for better pauses and to fit model limits.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split by major punctuation that warrants a pause
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If a single sentence is too long, split it by commas/semicolons
        if len(sentence) > max_chars:
            sub_sentences = re.split(r"(?<=[,;])\s+", sentence)
            for sub in sub_sentences:
                if len(current_chunk) + len(sub) < max_chars:
                    current_chunk += sub + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sub + " "
        elif len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def main():
    # CLI argument parsing
    parser = argparse.ArgumentParser(
        description="Generate voiceover audio from text using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voiceover_script.py -i article.txt -o voiceover_article.wav
  python voiceover_script.py -i script.txt -o output.wav -v my_voice.wav
  python voiceover_script.py -i content.txt  # auto-generates output name
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        default="article.txt",
        help="Input text file (default: article.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output WAV file (default: voiceover_<input_name>.wav)",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default="clone.wav",
        help="Voice reference WAV for cloning (default: clone.wav)",
    )
    args = parser.parse_args()

    # Configuration from CLI args
    INPUT_FILE = args.input
    AUDIO_PROMPT = args.voice

    # Auto-generate output filename if not provided
    if args.output:
        OUTPUT_FILE = args.output
    else:
        input_basename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        OUTPUT_FILE = f"voiceover_{input_basename}.wav"

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    #    device="cpu" #Just to test cpu speed

    print(f"Using device: {device}")

    # Load article
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        article_text = f.read()

    # Load model
    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # Prepare voice cloning
    if os.path.exists(AUDIO_PROMPT):
        print(f"Preparing voice cloning from {AUDIO_PROMPT}...")
        model.prepare_conditionals(AUDIO_PROMPT)
    else:
        print(f"Warning: {AUDIO_PROMPT} not found. Using default voice.")

    chunks = chunk_text(article_text)
    all_wavs = []

    print(f"Generating {len(chunks)} chunks...")

    # Silence tensor for padding between chunks (0.5 seconds for better pacing)
    silence = torch.zeros(1, int(model.sr * 0.5))

    for i, chunk in enumerate(chunks):
        # Concise logging for skill usage
        if os.getenv("CLAUDE_SKILL"):
            if i % 5 == 0 or i == len(chunks) - 1:
                print(f"Progress: {i + 1}/{len(chunks)}")
        else:
            print(f"Processing chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")

        try:
            # temperature=0.45: Lower temperature reduces variance, helping with accent consistency
            # repetition_penalty=1.1: Slight penalty to prevent stuttering/accent artifacts
            # seed_num=42: For reproducibility
            wav = model.generate(
                chunk, temperature=0.9, repetition_penalty=1.3, seed_num=42
            )

            # LUFS Normalization (replaces simple peak normalization)
            wav = norm_loudness(wav, model.sr, target_lufs=-19)

            all_wavs.append(wav)
            all_wavs.append(silence)
        except Exception as e:
            print(f"Error generating chunk {i + 1}: {e}")

    if not all_wavs:
        print("No audio generated.")
        return

    # Concatenate all chunks
    final_wav = torch.cat(all_wavs, dim=-1)

    # Final LUFS normalization to ensure consistent loudness across the whole file
    final_wav = norm_loudness(final_wav, model.sr, target_lufs=-19)

    # Save output using scipy (avoids torchcodec dependency)
    wav_np = final_wav.cpu().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.squeeze(0)
    # Convert to int16 for WAV file
    wav_int16 = (wav_np * 32767).astype(np.int16)
    wavfile.write(OUTPUT_FILE, model.sr, wav_int16)
    print(f"Saved voiceover to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
