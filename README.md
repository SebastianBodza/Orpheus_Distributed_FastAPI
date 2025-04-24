# Orpheus Distributed Streaming with FastAPI

A demo for running Orpheus streaming server detached. The SNAC and the LLM can be easily placed on different devices as the FastAPI uses an openAI compatible server. 

This is optimized for my finetune. Something is a bit off from the orignal implementation with the end tokens. 


> You might also want to take a look at the official implementation https://github.com/canopyai/Orpheus-TTS/tree/main/orpheus_tts_pypi/orpheus_tts and https://github.com/canopyai/Orpheus-TTS/tree/main/realtime_streaming_example 
> I unfortunately saw it after creating this implementation with Gemini 2.5. Difference to my implementation is the use of the AsnycLLMEngine from vllm directly, so detaching SNAC and the Orpheus LLM might be a bit harder esp. across different machines. Also they use a sliding window approach to mitigate the cracking sound at the end/beginning of each chunk. I am using a simple fading which also works good. 