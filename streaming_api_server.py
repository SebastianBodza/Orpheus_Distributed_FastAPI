import os
import torch
from snac import SNAC
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import asyncio
import functools
import struct
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import torch.nn.functional as F

# CHAANGE THE 
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://0.0.0.0:8000/v1") # port vLLM 8000 or 8001 / SGLang 30000
MODEL_NAME = os.environ.get("MODEL_NAME", "SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1")
SNAC_MODEL_NAME = os.environ.get("SNAC_MODEL_NAME", "hubertsiuzdak/snac_24khz")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "token123")

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 4000 
DEFAULT_REPETITION_PENALTY = 1.1
CODE_TOKEN_OFFSET = 128266
STOP_SEQUENCE = "<custom_token_2>"
AUDIO_SAMPLERATE = 24000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1


STREAM_CHUNK_SIZE_GROUPS = 30
INITIAL_CHUNK_SIZE_GROUPS = 3 

CODE_START_TOKEN_ID = 128257 
CODE_REMOVE_TOKEN_ID = 128258 
app = FastAPI()
SNAC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
snac_model = snac_model.to(SNAC_DEVICE)
if SNAC_DEVICE == "cuda":
    snac_model = snac_model.half()
snac_model.eval()
print(f"SNAC model loaded on {SNAC_DEVICE}")
print(f"Tokenizer loaded from {MODEL_NAME}")
print(f"Connected to vLLM at {VLLM_BASE_URL}")


class AudioRequest(BaseModel):
    text: str
    voice: str = "in_prompt"
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

def format_prompt_for_vllm_sync(prompt_text, voice="in_prompt"):
    """Synchronous version of formatting the text prompt."""
    if voice != "in_prompt" and voice != "":
        full_text = f"{voice}: {prompt_text}"
    else:
        full_text = prompt_text

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    decoded_text = tokenizer.decode(modified_input_ids[0], skip_special_tokens=False)
    return decoded_text

def tokenize_sync(text):
    """Synchronous tokenization."""
    return tokenizer.encode(text)


def redistribute_codes_sync(code_list):
    """Synchronous version of redistributing codes and decoding audio."""
    if not code_list:
        return torch.tensor([[]], device=SNAC_DEVICE, dtype=torch.float32)

    num_codes = len(code_list)
    num_groups = num_codes // 7
    if num_groups == 0:
         return torch.tensor([[]], device=SNAC_DEVICE, dtype=torch.float32)

    code_list = code_list[:num_groups * 7]

    layer_1, layer_2, layer_3 = [], [], []
    for i in range(num_groups):
        base_idx = 7 * i
        try:
            layer_1.append(code_list[base_idx])
            layer_2.append(code_list[base_idx + 1] - 4096)
            layer_3.append(code_list[base_idx + 2] - (2 * 4096))
            layer_3.append(code_list[base_idx + 3] - (3 * 4096))
            layer_2.append(code_list[base_idx + 4] - (4 * 4096))
            layer_3.append(code_list[base_idx + 5] - (5 * 4096))
            layer_3.append(code_list[base_idx + 6] - (6 * 4096))
        except IndexError:
            print(f"Warning: IndexError during code redistribution at group {i}. Skipping group.")
            break

    if not layer_1:
        return torch.tensor([[]], device=SNAC_DEVICE, dtype=torch.float32)

    codes = [
        torch.tensor(layer_1, device=SNAC_DEVICE).unsqueeze(0),
        torch.tensor(layer_2, device=SNAC_DEVICE).unsqueeze(0),
        torch.tensor(layer_3, device=SNAC_DEVICE).unsqueeze(0),
    ]

    with torch.no_grad():
        audio_hat = snac_model.decode(codes)
    return audio_hat

def apply_fade(audio_tensor, fade_samples):
    """Applies a fade-in and fade-out to the audio tensor."""
    if audio_tensor is None or audio_tensor.numel() < 2 * fade_samples:
        print("Warning: Audio tensor too short for fade. Returning original tensor.")
        return audio_tensor

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    num_samples = audio_tensor.shape[-1]
    fade_in = torch.linspace(0., 1., fade_samples, device=audio_tensor.device)
    fade_out = torch.linspace(1., 0., fade_samples, device=audio_tensor.device)
    audio_tensor[..., :fade_samples] *= fade_in
    audio_tensor[..., -fade_samples:] *= fade_out
    return audio_tensor.squeeze()

def convert_to_pcm16_bytes(audio_tensor, fade_ms=5):
    """Converts audio tensor to raw PCM 16-bit bytes with optional fade."""
    if audio_tensor is None or audio_tensor.numel() == 0:
        return b''

    if fade_ms > 0:
        fade_samples = int(AUDIO_SAMPLERATE * fade_ms / 1000)
        fade_samples = (fade_samples // 2) * 2
        if fade_samples > 0:
            audio_tensor = apply_fade(audio_tensor.detach(), fade_samples)

    audio_numpy = (audio_tensor.squeeze().cpu().to(torch.float32).numpy() * 32767)
    audio_numpy = np.clip(audio_numpy, -32768, 32767).astype(np.int16)
    return audio_numpy.tobytes()

def create_wav_header(sample_rate, bits_per_sample, channels, data_size=0xFFFFFFFF):
    """Creates a WAV header with potentially unknown data size for streaming."""
    riff_size = 36 + data_size
    if riff_size > 0xFFFFFFFF: riff_size = 0xFFFFFFFF

    header = b'RIFF'
    header += struct.pack('<I', riff_size)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', sample_rate * channels * bits_per_sample // 8)
    header += struct.pack('<H', channels * bits_per_sample // 8)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    return header


async def generate_audio_stream(request: AudioRequest):
    """
    Async generator that streams audio chunks for a given text prompt.
    Yields WAV header first, then raw PCM audio bytes.
    Yields the first chunk faster.
    """
    loop = asyncio.get_running_loop()

    try:
        formatted_prompt = await loop.run_in_executor(
            None, functools.partial(format_prompt_for_vllm_sync, request.text, request.voice)
        )
        print(f"Formatted Prompt: {formatted_prompt}")

        wav_header = create_wav_header(AUDIO_SAMPLERATE, AUDIO_BITS_PER_SAMPLE, AUDIO_CHANNELS)
        yield wav_header
        print("Yielded WAV header.")

        print(f"Starting vLLM stream for: '{request.text[:50]}...'")
        stream_kwargs = dict(
            model=MODEL_NAME,
            prompt=formatted_prompt,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=[STOP_SEQUENCE],
            stream=True,
            extra_body={'repetition_penalty': request.repetition_penalty},
        )
        response_stream = await client.completions.create(**stream_kwargs)

        accumulated_text = ""
        processed_code_count = 0
        start_token_found = False
        start_idx = -1
        first_chunk_yielded = False

        async for chunk in response_stream:
            if chunk.choices:
                chunk_text = chunk.choices[0].text or ""
                accumulated_text += chunk_text
                all_token_ids = await loop.run_in_executor(None, tokenize_sync, accumulated_text)

                if not start_token_found:
                    try:
                        start_idx = all_token_ids.index(CODE_START_TOKEN_ID)
                        start_token_found = True
                        print(f"Code start token ({CODE_START_TOKEN_ID}) found at index {start_idx}.")
                    except ValueError:
                        continue

                if start_token_found:
                    potential_code_tokens = all_token_ids[start_idx + 1:]

                    valid_raw_codes = [
                        token for token in potential_code_tokens
                        if token != CODE_REMOVE_TOKEN_ID and token >= CODE_TOKEN_OFFSET
                    ]

                    current_total_codes = len(valid_raw_codes)

                    if not first_chunk_yielded:
                        current_decode_chunk_size = INITIAL_CHUNK_SIZE_GROUPS * 7
                        print(f"Using initial chunk size: {current_decode_chunk_size} codes")
                    else:
                        current_decode_chunk_size = STREAM_CHUNK_SIZE_GROUPS * 7

                    if current_total_codes >= processed_code_count + current_decode_chunk_size:
                        codes_to_process_now_count = ( (current_total_codes - processed_code_count) // current_decode_chunk_size ) * current_decode_chunk_size
                        end_process_idx = processed_code_count + codes_to_process_now_count

                        if end_process_idx > processed_code_count:
                            codes_to_process_raw = valid_raw_codes[processed_code_count : end_process_idx]
                            print(f"Processing codes from {processed_code_count} to {end_process_idx} ({len(codes_to_process_raw)} codes)")

                            codes_to_process = [t - CODE_TOKEN_OFFSET for t in codes_to_process_raw]

                            audio_hat = await loop.run_in_executor(
                                None, redistribute_codes_sync, codes_to_process
                            )

                            pcm_bytes = convert_to_pcm16_bytes(audio_hat, fade_ms=50) # Apply fade here
                            if pcm_bytes:
                                print(f"Yielding {len(pcm_bytes)} bytes of audio data.")
                                yield pcm_bytes
                                first_chunk_yielded = True 
                                print("Warning: No PCM bytes generated for this chunk.")


                            processed_code_count = end_process_idx

        print("Stream finished. Processing remaining codes.")
        all_token_ids = await loop.run_in_executor(None, tokenize_sync, accumulated_text)

        if start_token_found:
            potential_code_tokens = all_token_ids[start_idx + 1:]
            valid_raw_codes = [
                token for token in potential_code_tokens
                if token != CODE_REMOVE_TOKEN_ID and token >= CODE_TOKEN_OFFSET
            ]
            current_total_codes = len(valid_raw_codes)

            if current_total_codes > processed_code_count:
                remaining_codes_raw = valid_raw_codes[processed_code_count:]
                num_remaining = len(remaining_codes_raw)
                final_len = (num_remaining // 7) * 7

                if final_len > 0:
                    codes_to_process = [t - CODE_TOKEN_OFFSET for t in remaining_codes_raw[:final_len]]
                    print(f"Processing final {len(codes_to_process)} codes.")

                    audio_hat = await loop.run_in_executor(
                        None, redistribute_codes_sync, codes_to_process
                    )
                    pcm_bytes = convert_to_pcm16_bytes(audio_hat, fade_ms=50)
                    if pcm_bytes:
                        print(f"Yielding final {len(pcm_bytes)} bytes of audio data.")
                        yield pcm_bytes
        else:
             print("Warning: Code start token never found in the entire response.")


        print("Audio stream generation complete.")

    except asyncio.CancelledError:
        print("Stream cancelled by client.")
    except Exception as e:
        print(f"Error during audio stream generation: {e}")
        import traceback
        traceback.print_exc()


@app.post("/generate-audio-stream/")
async def generate_audio_stream_endpoint(request: AudioRequest):
    """
    Streams generated audio as a WAV file.
    """
    print(f"Received streaming request for: '{request.text[:50]}...'")
    return StreamingResponse(
        generate_audio_stream(request),
        media_type="audio/wav"
    )

@app.get("/")
async def read_root():
    return {"message": "SNAC + vLLM/sglang Streaming Audio Generation API is running."}

if __name__ == "__main__":
    print("Starting FastAPI server for streaming...")
    uvicorn.run("streaming_api_server:app", host="0.0.0.0", port=8001, reload=False)

