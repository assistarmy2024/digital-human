# riva_asr_client.py
import os
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters
from typing import List, Optional, Tuple, Union


def get_transcription(
    input_file: str,
    uri: Optional[Union[str, os.PathLike]] = None,
    ssl_cert: str = None,
    language_code: str = "en-US",
    model_name: str = "conformer",
    automatic_punctuation: bool = True,
    show_intermediate: bool = False,
    boosted_lm_words: list = None,
    boosted_lm_score: float = 4.0
) -> str:
    """
    Function to get ASR transcription from an audio file using Riva AI.
    
    Args:
        input_file (str): Path to the input audio file.
        server (str): Riva ASR server address.
        ssl_cert (str, optional): SSL certificate for secure connection.
        language_code (str, optional): Language code for ASR.
        model_name (str, optional): ASR model name.
        automatic_punctuation (bool, optional): Enable automatic punctuation.
        show_intermediate (bool, optional): Show intermediate transcripts.
        boosted_lm_words (list, optional): Words to boost in ASR.
        boosted_lm_score (float, optional): Boost score for boosted words.

    Returns:
        str: Final transcription text.
    """
    if not os.path.isfile(input_file):
        raise ValueError(f"Invalid input file path: {input_file}")

    auth = riva.client.Auth(ssl_cert=ssl_cert, use_ssl=False, uri=uri)
    asr_service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code=language_code,
            model=model_name,
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=automatic_punctuation,
            verbatim_transcripts=True,
        ),
        interim_results=show_intermediate,
    )

    if boosted_lm_words:
        riva.client.add_word_boosting_to_config(config, boosted_lm_words, boosted_lm_score)

    transcription = []
    with riva.client.AudioChunkFileIterator(input_file, chunk_n_frames=1600) as audio_chunk_iterator:
        responses = asr_service.streaming_response_generator(audio_chunks=audio_chunk_iterator, streaming_config=config)
        for response in responses:
            for result in response.results:
                if result.is_final:
                    transcription.append(result.alternatives[0].transcript)

    return " ".join(transcription).strip()

