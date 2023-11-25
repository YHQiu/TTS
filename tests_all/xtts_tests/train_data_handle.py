import concurrent.futures
import os
import csv
import librosa
import numpy as np
import soundfile as sf
import argparse
from praatio import textgrid
from tqdm import tqdm


def convert_flac_to_wav(flac_file, output_path):
    y, sr = librosa.load(flac_file, sr=None)
    sf.write(output_path, y, sr)

def process_entry(entrie, y, output_folder, wav_file_name, sr, gain=1.5):
    start_sample = int(entrie.start * sr)
    end_sample = int(entrie.end * sr)
    audio_segment = y[start_sample:end_sample]

    # Adjust the volume by applying gain
    audio_segment *= gain

    # Clip the audio to ensure it stays within the valid range [-1, 1]
    audio_segment = np.clip(audio_segment, -1.0, 1.0)

    segment_output_path = os.path.join(output_folder, 'wavs', f"{wav_file_name.split('.')[0]}_{start_sample}_{end_sample}.wav")
    sf.write(segment_output_path, audio_segment, sr, format="WAV")

    return f"{wav_file_name.split('.')[0]}_{start_sample}_{end_sample}", entrie.label


def process_data(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, 'wavs')):
        os.makedirs(os.path.join(output_folder, 'wavs'))

    processed_records_file = os.path.join(output_folder, 'processed_records.txt')
    processed_records = set()

    if os.path.exists(processed_records_file):
        with open(processed_records_file, 'r') as f:
            processed_records = set(f.read().splitlines())

    new_records = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(input_folder):
            for file in tqdm(files, desc='Processing files', unit='file'):
                futures = []
                if file.endswith('.flac') and file not in processed_records:

                    flac_path = os.path.join(root, file)
                    wav_file_name = file.replace('.flac', '.wav')

                    text_file = file.replace('.flac', '.TextGrid')
                    text_path = os.path.join(root, 'TextGrid', text_file)
                    text_path = text_path.replace("/wav", "")
                    text_path = text_path.replace("\\wav", "")

                    y, sr = librosa.load(flac_path, sr=None)

                    # Read TextGrid file to extract intervals
                    tg = textgrid.openTextgrid(text_path, False)
                    # 获取标注层信息
                    tiers = tg.tiers
                    for tier in tqdm(tiers, desc='Processing tiers', unit='tier'):
                        max1 = 1
                        for entrie in tqdm(tier.entries, desc='Processing entries', unit='entrie'):
                            if max1 < 0: break
                            futures.append(executor.submit(process_entry, entrie, y, output_folder, wav_file_name, sr))
                            max1 -= 1
                        if max1 < 0: break

                    processed_records.add(file)

                for future in concurrent.futures.as_completed(futures):
                    new_records.append(future.result())

    print("start write processed_records_file")
    with open(processed_records_file, 'a') as f:
        for record in processed_records:
            f.write(record + '\n')
    print("end write processed_records_file")

    print("start write csvfile")
    with open(os.path.join(output_folder, 'train_100_data/metadata.csv'), 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')
        for record in new_records:
            csv_writer.writerow(record)
    print("end write csvfile")

def main():
    parser = argparse.ArgumentParser(description='Process audio and text data')
    parser.add_argument('--input-path', required=False, default="train_L/", help='data/dataset/train_L')
    parser.add_argument('--output-path', required=False, default="train_data/", help='"../train_outputs/data/train_data')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    process_data(input_path, output_path)


if __name__ == "__main__":
    main()
