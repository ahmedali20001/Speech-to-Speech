# STT-NLP-TTS Microservice

This is an end-to-end STT (Speech-to-Text) → NLP (Intent/Entity Recognition) → TTS (Text-to-Speech) microservice. It ingests 16-bit PCM mono 8 kHz audio chunks via WebSocket, transcribes them using Whisper, performs intent and entity recognition using Transformers, and streams back TTS responses.

## Features
- **WebSocket**: Handles audio input/output at `ws://<host>:8765`.
- **Health Endpoint**: Check service status at `http://<host>:8080/health`.
- **Structured Logs**: JSON logs with timestamps and request IDs, stored in `logs/server.log`.
- **Latency Tracking**: Measures p95 latency for processing requests.
- **File Storage**: Saves input (`audio/record_chunk.wav`) and output (`audio/tts_output_gtts.wav`) audio files.
- **Dockerized**: Reproducible builds with pre-cached models.

## Setup

### Prerequisites
- Docker (for containerized deployment)
- Web browser (for client testing)
- Optional: Python 3.10+ (for local development)

### Local Development
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd stt-nlp-tts
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   python server.py
   ```
5. Open `index.html` in a browser (e.g., via `python3 -m http.server 8000` and navigate to `http://localhost:8000`).

### Docker Deployment
1. Build the Docker image:
   ```bash
   docker build --platform linux/arm64 -t stt-nlp-tts .
   ```
   Note: Use `--platform linux/amd64` for non-Apple hardware.
2. Run the container:
   ```bash
   docker run -p 8765:8765 -p 8080:8080 -v $(pwd)/audio:/app/audio -v $(pwd)/logs:/app/logs stt-nlp-tts
   ```
   - Maps ports 8765 (WebSocket) and 8080 (HTTP).
   - Persists `audio/` and `logs/` directories to the host.

### Usage
1. Open `index.html` in a browser.
2. Read the note: "This is currently built to recognize greetings, so say things like, e.g., 'Hi! How are you?'"
3. Click "Start Recording", speak a greeting, and click "Stop Recording".
4. The server processes the audio, saves input/output files, and streams back a TTS response.
5. Check the health endpoint: `curl http://localhost:8080/health`.
6. View logs in `logs/server.log` for structured JSON output.

## Configuration Knobs
Set these via environment variables when running the server or Docker container:

| Variable              | Description                              | Default                          |
|-----------------------|------------------------------------------|----------------------------------|
| `SAMPLE_RATE`         | Audio sample rate (Hz)                  | 8000                             |
| `CHANNELS`            | Number of audio channels                | 1                                |
| `DTYPE`               | Audio data type                         | int16                            |
| `STT_MODEL`           | Speech-to-text model                    | openai/whisper-tiny              |
| `INTENT_MODEL`        | Intent classification model             | distilbert-base-uncased-finetuned-sst-2-english |
| `NER_MODEL`           | Entity recognition model                | dslim/bert-base-NER              |
| `SENTIMENT_THRESHOLD` | Sentiment score threshold for greetings  | 0.7                              |
| `WEBSOCKET_PORT`      | WebSocket port                          | 8765                             |
| `HTTP_PORT`           | HTTP health endpoint port               | 8080                             |
| `LOG_FILE`            | Path to log file                        | logs/server.log                  |

Example (Docker):
```bash
docker run -e STT_MODEL=openai/whisper-base -e SENTIMENT_THRESHOLD=0.8 -p 8765:8765 -p 8080:8080 stt-nlp-tts
```

## Logs
- Stored in `logs/server.log` (rotated at 10MB, 5 backups).
- JSON format with `timestamp`, `level`, `request_id`, and `message` (e.g., transcription, latency).
- Example:
  ```json
  {"timestamp": "2025-09-09 18:42:00,123", "level": "INFO", "request_id": "uuid", "message": "Transcribed: Hello! How are you?"}
  ```

## Health Endpoint
- URL: `http://<host>:8080/health`
- Response:
  ```json
  {
    "status": "healthy",
    "uptime": 123.456,
    "models_loaded": true,
    "p95_latency": 2.345
  }
  ```

## Latency
- p95 latency is computed every 10 requests and logged.
- Available in the health endpoint (`p95_latency` in seconds).

## Notes
- Models are cached in the Docker image for offline use.
- Audio files are saved in `audio/` (overwritten per request). To persist multiple files, modify `server.py` to append timestamps.
- For production, use `wss://` and secure the health endpoint.
- Tested on macOS (M2, MPS) and Linux (x86).