from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import riva.client
import requests
import tempfile
import wave
from riva.client.proto.riva_audio_pb2 import AudioEncoding
from pydub import AudioSegment
from riva_asr_client import get_transcription

import re

# NVIDIA Riva and LLaMA API configurations
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
RIVA_AUTH_HEADER = [
    ["authorization", "Bearer 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"],
    ["function-id", "1598d209-5e27-4d3c-8079-4751568b1081"],
]
RIVA_AUTH_HEADER_TTS = [
    ["authorization", "Bearer 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"],
    ["function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972"],
]
LLM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_API_KEY = "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"

# Initialize FastAPI app
app = FastAPI()

# Function for Riva ASR (speech-to-text)
def riva_asr_transcription(audio_path: str) -> str:
    if not os.path.isfile(audio_path):
        raise HTTPException(status_code=400, detail=f"Invalid file path: {audio_path}")

    auth = riva.client.Auth(use_ssl=True, uri=RIVA_SERVER, metadata_args=RIVA_AUTH_HEADER)
    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
        ),
        interim_results=False,  # Disable interim results to avoid partial duplicates
    )

    transcription_list = []

    # Streaming ASR to get transcriptions
    with riva.client.AudioChunkFileIterator(audio_path, chunk_n_frames=1600) as audio_chunk_iterator:
        try:
            responses = asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator, streaming_config=config
            )

            for response in responses:
                for result in response.results:
                    if result.is_final:  # Only collect final results
                        transcript = result.alternatives[0].transcript
                        transcription_list.append(transcript)

            full_transcription = " ".join(transcription_list).strip()

            # Clean up any repeated phrases using `remove_repeated_phrases`
            cleaned_transcription = remove_repeated_phrases(full_transcription)

            print("Cleaned ASR Transcription:", cleaned_transcription)
            return cleaned_transcription

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during ASR transcription: {e}")

# Function to query LLaMA LLM (text generation)
def llama_query_llm(user_input: str) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [{"role": "user", "content": user_input}],
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 100,
        "stream": False
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        llm_response = response.json()
        answer = llm_response["choices"][0]["message"]["content"]
        print("LLMAnswer",answer)
        return answer.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM query: {e}")

# Function for Riva TTS (text-to-speech)
def riva_tts_generate(text: str, output_path: str) -> str:
    auth = riva.client.Auth(uri=RIVA_SERVER, use_ssl=True, metadata_args=RIVA_AUTH_HEADER_TTS)
    tts_service = riva.client.SpeechSynthesisService(auth)

    try:
        print("Generating TTS response...")

        # Specify a valid voice manually (adjust based on your deployment)
        response = tts_service.synthesize(
            text=text,
            voice_name="English-US.Male-1",  # Replace with a valid voice name
            language_code="en-US",
            encoding= AudioEncoding.LINEAR_PCM,
            sample_rate_hz=44100,
        )

        # Save the audio to output file
        with wave.open(output_path, "wb") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2)  # 16-bit PCM
            out_f.setframerate(44100)
            out_f.writeframes(response.audio)

        print(f"TTS audio saved to {output_path}")
        return output_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during TTS synthesis: {e}")


def remove_repeated_phrases(transcription: str) -> str:
    # Remove repeated words or phrases
    cleaned_text = re.sub(r'(\b\w+\b)( \1)+', r'\1', transcription, flags=re.IGNORECASE)
    return cleaned_text

# Endpoint to process audio, perform ASR, get LLM response, and return synthesized audio
@app.post("/process-audio/")
async def process_audio(filepath: str):
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Convert the audio to mono if necessary
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_mono_file:
            mono_filepath = temp_mono_file.name

        # Load the audio file with pydub and ensure it is mono
        audio = AudioSegment.from_file(filepath)
        if audio.channels > 1:
            audio = audio.set_channels(1)  # Convert to mono
        audio.export(mono_filepath, format="wav")  # Save as a temporary mono file

        # Perform ASR to get user input text
        transcription = riva_asr_transcription(mono_filepath)
        cleaned_transcription = remove_repeated_phrases(transcription)
        print("Cleaned Transcription:", cleaned_transcription)

        # Get LLM response for the cleaned transcription
        llm_response = llama_query_llm(cleaned_transcription)

        # Generate TTS for the LLM response
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
            #temp_output_path = temp_output.name
            #print("temp_output_path----------->",temp_output_path)
        temp_output_path = "C:/Users/iassi/Downloads/KairosSample-UE5-Source-CL233291/Saved/BouncedWavFiles/ACERecording.wav"
        synthesized_audio_path = riva_tts_generate(llm_response, temp_output_path)

        # Clean up temporary mono file
        os.remove(mono_filepath)

        return {
            "transcription": cleaned_transcription,
            "llm_response": llm_response,
            "audio_response": FileResponse(synthesized_audio_path, media_type="audio/wav"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))