from pydub import AudioSegment

# Load the stereo audio file
audio = AudioSegment.from_wav(r"C:\Users\iassi\Documents\Sound Recordings\Recording_1.wav")

# Convert to mono
mono_audio = audio.set_channels(1)

# Export the mono file
mono_audio.export(r"C:\Users\iassi\Documents\Sound Recordings\Recording_1_mono.wav", format="wav")
