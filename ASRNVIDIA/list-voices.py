import riva.client
from riva.client.proto.riva_tts_pb2 import RivaSynthesisConfigRequest

def list_available_voices():
    auth = riva.client.Auth(uri="grpc.nvcf.nvidia.com:443", use_ssl=True, metadata_args=[
        ["authorization", "Bearer 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"],
        ["function-id", "1598d209-5e27-4d3c-8079-4751568b1081"],
    ])
    tts_service = riva.client.SpeechSynthesisService(auth)
    config_request = RivaSynthesisConfigRequest()
    response = tts_service.stub.GetRivaSynthesisConfig(config_request)
    print
    for model_config in response.model_config:
        language = model_config.parameters["language_code"]
        voice_name = model_config.parameters.get("voice_name", "Unnamed")
        print(f"Language: {language}, Voice: {voice_name}")

list_available_voices()
