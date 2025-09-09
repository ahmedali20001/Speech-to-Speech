# Speech-to-Speech
This is an end-to-end STT (Speech-to-Text) → NLP (Intent/Entity Recognition) → TTS (Text-to-Speech) microservice. It ingests 16-bit PCM mono 8 kHz audio chunks via WebSocket, transcribes them using Whisper, performs intent and entity recognition using Transformers, and streams back TTS responses.
