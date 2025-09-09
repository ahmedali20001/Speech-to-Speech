import asyncio
import websockets
import numpy as np
import soundfile as sf
import os
import time
import json
import logging
import logging.handlers
from aiohttp import web
from transformers import pipeline
from pydub import AudioSegment
import librosa
from gtts import gTTS
import re
from statistics import quantiles
import uuid
from typing import List
import warnings
from transformers.utils import is_torch_available

# Suppress transformers warnings (optional, to clean up output)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
if is_torch_available():
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cpu"

# Configuration (configurable via environment variables or defaults)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 8000))
CHANNELS = int(os.getenv("CHANNELS", 1))
DTYPE = os.getenv("DTYPE", "int16")
STT_MODEL = os.getenv("STT_MODEL", "openai/whisper-tiny")
INTENT_MODEL = os.getenv("INTENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
NER_MODEL = os.getenv("NER_MODEL", "dslim/bert-base-NER")
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", 0.7))
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8765))
HTTP_PORT = int(os.getenv("HTTP_PORT", 8080))
LOG_FILE = os.getenv("LOG_FILE", "logs/server.log")

# Ensure directories exist
os.makedirs("audio", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Custom JSON Formatter for structured logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "request_id": getattr(record, 'request_id', 'N/A'),
            "message": record.getMessage()
        }
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)

# Structured logging setup
logger = logging.getLogger("STT_NLP_TTS")
logger.setLevel(logging.INFO)
formatter = JsonFormatter()

file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Latency tracking
latencies: List[float] = []

# Preload pipelines with device support
logger.info('Initializing pipelines', extra={'request_id': 'startup', 'extra_data': {}})
asr_pipeline = pipeline("automatic-speech-recognition", model=STT_MODEL, device=device)
intent_classifier = pipeline("text-classification", model=INTENT_MODEL, device=device)
ner_pipeline = pipeline("ner", model=NER_MODEL, device=device)
logger.info('Pipelines initialized', extra={'request_id': 'startup', 'extra_data': {}})

def detect_intent(text: str, request_id: str) -> bool:
    text = text.strip().lower()
    greeting_patterns = [
        r"hello", r"hi", r"hey", r"how are you", r"greetings", r"good morning",
        r"good afternoon", r"good evening"
    ]
    is_greeting = any(re.search(pattern, text) for pattern in greeting_patterns)
    sentiment = intent_classifier(text)[0]
    is_positive = sentiment['label'] == 'POSITIVE' and sentiment['score'] > SENTIMENT_THRESHOLD
    logger.info(f'Intent detection: is_greeting={is_greeting}, sentiment={sentiment}', extra={'request_id': request_id, 'extra_data': {'text': text}})
    return is_greeting and is_positive

def extract_entities(text: str, request_id: str) -> list:
    entities = ner_pipeline(text)
    result = [{"word": e['word'], "entity": e['entity'], "score": e['score']} for e in entities]
    logger.info(f'Entity extraction: {result}', extra={'request_id': request_id, 'extra_data': {}})
    return result

def generate_response(is_greeting: bool, entities: list, request_id: str) -> str:
    response = "I'm fine, thank you!" if is_greeting else "I understood your input, but I'm not sure how to respond. Could you clarify?"
    logger.info(f'Response generated: {response}', extra={'request_id': request_id, 'extra_data': {}})
    return response

async def handle_connection(websocket):
    request_id = str(uuid.uuid4())
    logger.info('Client connected', extra={'request_id': request_id, 'extra_data': {}})
    start_time = time.time()
    try:
        # Receive audio data
        audio_chunks = []
        while True:
            data = await websocket.recv()
            if isinstance(data, bytes):
                chunk = np.frombuffer(data, dtype=np.int16)
                audio_chunks.append(chunk)
            elif data == "END":
                break

        if not audio_chunks:
            logger.warning('No audio received', extra={'request_id': request_id, 'extra_data': {}})
            await websocket.send("No audio received")
            return

        # Combine chunks and save
        audio = np.concatenate(audio_chunks, axis=0)
        audio_file = "audio/record_chunk.wav"
        sf.write(audio_file, audio, SAMPLE_RATE)
        logger.info(f'Input audio saved to {audio_file}', extra={'request_id': request_id, 'extra_data': {}})

        # Step 1: Speech-to-Text
        res = asr_pipeline(audio_file)
        transcribed_text = res.get('text', '').strip()
        logger.info(f'Speech-to-text: {transcribed_text}', extra={'request_id': request_id, 'extra_data': {}})

        # Step 2: Intent Detection
        is_greeting = detect_intent(transcribed_text, request_id)

        # Step 3: Entity Extraction
        entities = extract_entities(transcribed_text, request_id)

        # Step 4: Generate Response
        response = generate_response(is_greeting, entities, request_id)

        # Step 5: Text-to-Speech
        tts = gTTS(response, lang='en')
        temp_mp3 = "temp/temp_tts_output.mp3"
        tts.save(temp_mp3)
        audio_data, sr = librosa.load(temp_mp3, sr=SAMPLE_RATE, mono=True)
        audio_data_16bit = np.int16(audio_data * 32767)
        tts_file = "audio/tts_output_gtts.wav"
        sf.write(tts_file, audio_data_16bit, SAMPLE_RATE)
        logger.info(f'TTS output saved to {tts_file}', extra={'request_id': request_id, 'extra_data': {}})

        # Send TTS audio back to client
        with open(tts_file, 'rb') as f:
            await websocket.send(f.read())

        # Log latency
        latency = time.time() - start_time
        latencies.append(latency)
        if len(latencies) >= 10:  # Compute p95 every 10 requests
            p95 = quantiles(latencies, n=100)[94]  # 95th percentile
            logger.info(f'Latency stats: p95={p95}s, sample_count={len(latencies)}', extra={'request_id': request_id, 'extra_data': {}})
            # Reset periodically to avoid memory growth (optional)
            if len(latencies) > 1000:
                latencies.clear()

    except Exception as e:
        logger.error(f'Error: {str(e)}', extra={'request_id': request_id, 'extra_data': {}})
        await websocket.send(f"Error: {str(e)}")

async def health_check(request):
    request_id = str(uuid.uuid4())
    status = {
        "status": "healthy",
        "uptime": time.time() - server_start_time,
        "models_loaded": all([asr_pipeline, intent_classifier, ner_pipeline]),
        "device": device,
        "p95_latency": quantiles(latencies, n=100)[94] if latencies else None
    }
    logger.info('Health check', extra={'request_id': request_id, 'extra_data': {'status': status}})
    return web.json_response(status)

# HTTP server for health endpoint
app = web.Application()
app.add_routes([web.get('/health', health_check)])

# Global start time for uptime
server_start_time = time.time()

# Start both WebSocket and HTTP servers
async def main():
    # Start WebSocket server
    ws_server = await websockets.serve(handle_connection, "0.0.0.0", WEBSOCKET_PORT)
    logger.info(f'WebSocket server started on ws://0.0.0.0:{WEBSOCKET_PORT}', extra={'request_id': 'startup', 'extra_data': {}})

    # Start HTTP server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', HTTP_PORT)
    await site.start()
    logger.info(f'HTTP server started on http://0.0.0.0:{HTTP_PORT}', extra={'request_id': 'startup', 'extra_data': {}})

    await asyncio.gather(ws_server.wait_closed(), runner.cleanup())

if __name__ == "__main__":
    asyncio.run(main())