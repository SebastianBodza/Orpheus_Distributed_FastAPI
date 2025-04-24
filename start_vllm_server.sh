CUDA_VISIBLE_DEVICES=0 vllm serve  SebastianBodza/Kartoffel_Orpheus-3B_german_natural-v0.1 --dtype auto --enable-chunked-prefill --enable-prefix-caching --max_model_len 8000 --quantization fp8
