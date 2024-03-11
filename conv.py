from pydub import AudioSegment
import os

def convert_wav_to_mp3(input_wav):
    # Extracting the file name and extension from the input path
    base_name, _ = os.path.splitext(os.path.basename(input_wav))

    # Creating the output MP3 file name based on the input WAV file name
    output_mp3 = f"{base_name}.mp3"

    # Load the WAV file
    audio = AudioSegment.from_wav(input_wav)

    # Export the audio to MP3
    audio.export(output_mp3, format="mp3")
