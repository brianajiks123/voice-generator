import argparse
import io
import json
import logging
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen
from flask import Flask, request, jsonify
from waitress import serve
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import VOICES_JSON, download_voice

_LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Piper TTS HTTP Server (production dengan Waitress)")
    
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=6969, help="HTTP server port")
    
    parser.add_argument("-m", "--model", required=True, help="Path atau nama voice (contoh: id_ID-news_tts-medium)")
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    
    parser.add_argument("--length-scale", "--length_scale", type=float, help="Phoneme length")
    parser.add_argument("--noise-scale", "--noise_scale", type=float, help="Generator noise")
    parser.add_argument("--noise-w-scale", "--noise_w_scale", "--noise-w", "--noise_w", type=float, help="Phoneme width noise")
    
    parser.add_argument("--cuda", action="store_true", help="Use GPU (jika tersedia)")
    
    parser.add_argument("--sentence-silence", "--sentence_silence", type=float, default=0.0, help="Seconds of silence after each sentence")
    
    parser.add_argument("--data-dir", "--data_dir", action="append", default=[str(Path.cwd())], help="Data directory untuk model (default: current dir)")
    parser.add_argument("--download-dir", "--download_dir", help="Path untuk download voices (default: first data dir)")
    
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages")
    
    parser.add_argument("--threads", type=int, default=8, help="Jumlah threads Waitress (default: 8)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug("Arguments: %s", args)
    
    if not args.download_dir:
        args.download_dir = args.data_dir[0]
    
    download_dir = Path(args.download_dir)
    
    model_path = Path(args.model)
    if not model_path.exists():
        voice_name = args.model
        for data_dir in args.data_dir:
            maybe_model_path = Path(data_dir) / f"{voice_name}.onnx"
            _LOGGER.debug("Mencari model di: %s", maybe_model_path)
            if maybe_model_path.exists():
                model_path = maybe_model_path
                break
    
    if not model_path.exists():
        _LOGGER.error("Model tidak ditemukan: %s", args.model)
        sys.exit(1)
    
    default_model_id = model_path.name.rstrip(".onnx")
    _LOGGER.info("Using model: %s (ID: %s)", model_path, default_model_id)
    
    # Load voice default
    default_voice = PiperVoice.load(model_path, use_cuda=args.cuda)
    loaded_voices: Dict[str, PiperVoice] = {default_model_id: default_voice}
    
    app = Flask(__name__)
    
    @app.route("/voices", methods=["GET"])
    def app_voices() -> Dict[str, Any]:
        voices_dict: Dict[str, Any] = {}
        config_paths: List[Path] = [Path(f"{model_path}.json")]
        
        for data_dir in args.data_dir:
            for onnx_path in Path(data_dir).glob("*.onnx"):
                config_path = Path(f"{onnx_path}.json")
                if config_path.exists():
                    config_paths.append(config_path)
        
        for config_path in config_paths:
            model_id = config_path.name.rstrip(".onnx.json")
            if model_id in voices_dict:
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    voices_dict[model_id] = json.load(f)
            except Exception as e:
                _LOGGER.warning("Failed load config %s: %s", config_path, e)
        
        return voices_dict
    
    @app.route("/all-voices", methods=["GET"])
    def app_all_voices() -> Dict[str, Any]:
        try:
            with urlopen(VOICES_JSON) as response:
                return json.load(response)
        except Exception as e:
            _LOGGER.error("Failed fetch all voices: %s", e)
            return {"error": str(e)}, 500
    
    @app.route("/download", methods=["POST"])
    def app_download() -> str:
        try:
            data = request.get_json()
            model_id = data.get("voice")
            if not model_id:
                return jsonify({"error": "Parameter 'voice' is must"}), 400
            
            force_redownload = data.get("force_redownload", False)
            downloaded = download_voice(model_id, download_dir, force_redownload=force_redownload)
            return downloaded
        except Exception as e:
            _LOGGER.error("Failed download: %s", e)
            return jsonify({"error": str(e)}), 500
    
    @app.route("/", methods=["POST"])
    def app_synthesize():
        try:
            data = request.get_json()
            text = data.get("text", "").strip()
            if not text:
                return jsonify({"error": "Text is not null"}), 400
            
            _LOGGER.debug("Request synthesize: %s", data)
            
            model_id = data.get("voice", default_model_id)
            voice = loaded_voices.get(model_id)
            
            if voice is None:
                for data_dir in args.data_dir:
                    maybe_path = Path(data_dir) / f"{model_id}.onnx"
                    if maybe_path.exists():
                        _LOGGER.info("Load new voice: %s", model_id)
                        voice = PiperVoice.load(maybe_path, use_cuda=args.cuda)
                        loaded_voices[model_id] = voice
                        break
            
            if voice is None:
                _LOGGER.warning("Voice %s is not found, using default", model_id)
                voice = default_voice
            
            # Speaker handling
            speaker_id: Optional[int] = data.get("speaker_id")
            if speaker_id is None and voice.config.num_speakers > 1:
                speaker = data.get("speaker")
                if speaker:
                    speaker_id = voice.config.speaker_id_map.get(speaker)
                if speaker_id is None:
                    speaker_id = args.speaker or 0
            
            if speaker_id is not None and speaker_id >= voice.config.num_speakers:
                speaker_id = 0
            
            # Synthesis config
            syn_config = SynthesisConfig(
                speaker_id=speaker_id,
                length_scale=float(
                    data.get("length_scale", args.length_scale if args.length_scale is not None else voice.config.length_scale)
                ),
                noise_scale=float(
                    data.get("noise_scale", args.noise_scale if args.noise_scale is not None else voice.config.noise_scale)
                ),
                noise_w_scale=float(
                    data.get("noise_w_scale", args.noise_w_scale if args.noise_w_scale is not None else voice.config.noise_w_scale)
                ),
            )
            
            _LOGGER.debug("Synthesize '%s' with config: %s", text, syn_config)
            
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wav_file:
                    wav_params_set = False
                    for i, chunk in enumerate(voice.synthesize(text, syn_config)):
                        if not wav_params_set:
                            wav_file.setframerate(chunk.sample_rate)
                            wav_file.setsampwidth(chunk.sample_width)
                            wav_file.setnchannels(chunk.sample_channels)
                            wav_params_set = True
                        
                        if i > 0:
                            silence_bytes = int(voice.config.sample_rate * args.sentence_silence * 2)
                            wav_file.writeframes(bytes(silence_bytes))
                        
                        wav_file.writeframes(chunk.audio_int16_bytes)
                
                wav_io.seek(0)
                return wav_io.read(), 200, {"Content-Type": "audio/wav"}
        
        except Exception as e:
            _LOGGER.exception("Error when synthesize")
            return jsonify({"error": str(e)}), 500

    print(f"Starting Voice Generate Server")
    print(f"  Host   : {args.host}")
    print(f"  Port   : {args.port}")
    print(f"  Threads: {args.threads}")
    print(f"  Model  : {args.model}")
    
    serve(
        app,
        host=args.host,
        port=args.port,
        threads=args.threads,
    )

if __name__ == "__main__":
    main()
