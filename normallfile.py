from pydub import AudioSegment
import os

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def process_folder(input_folder, output_folder, target_dBFS=-30.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"normalized_{filename}")

            sound = AudioSegment.from_file(input_path, "wav")
            normalized_sound = match_target_amplitude(sound, target_dBFS)
            normalized_sound.export(output_path, format="wav")

if __name__ == "__main__":
    input_folder_path = "data_noise"
    output_folder_path = "data_noise-30"
    process_folder(input_folder_path, output_folder_path)