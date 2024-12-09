import os
from pydub import AudioSegment

# Constants
DATA_DIR = "clone_model\opencv\.idea\stuff_to_enhance_model\crowd-worried-90368.mp3"  # Directory containing .aac files
OUTPUT_DIR = "clone_model\opencv\.idea\stuff_to_enhance_model"  # Directory to save re-encoded .wav files
TARGET_SAMPLE_RATE = 16000  # Target sample rate in Hz
TARGET_BIT_DEPTH = 16  # Target bit depth

def reencode_aac_to_wav(input_dir, output_dir, sample_rate=TARGET_SAMPLE_RATE, bit_depth=TARGET_BIT_DEPTH):
    """
    Converts all .aac files in the input directory to .wav with specified parameters.
    
    Args:
        input_dir (str): Path to the input directory containing .aac files.
        output_dir (str): Path to the output directory to save .wav files.
        sample_rate (int): Desired sample rate in Hz.
        bit_depth (int): Desired bit depth.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".aac"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path_dir = os.path.join(output_dir, relative_path)
                output_file_name = os.path.splitext(file)[0] + ".wav"
                output_path = os.path.join(output_path_dir, output_file_name)
                
                if not os.path.exists(output_path_dir):
                    os.makedirs(output_path_dir)

                try:
                    print(f"Processing {input_path}...")
                    audio = AudioSegment.from_file(input_path, format="aac")
                    audio = audio.set_frame_rate(sample_rate).set_sample_width(bit_depth // 8).set_channels(1)
                    audio.export(output_path, format="wav")
                    print(f"Saved to {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# Main function
if __name__ == "__main__":
    reencode_aac_to_wav(DATA_DIR, OUTPUT_DIR)
    print(f"Re-encoding complete. Files saved in {OUTPUT_DIR}")
