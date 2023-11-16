import os
import csv
import librosa
import soundfile as sf
import argparse


def convert_flac_to_wav(flac_file, output_path):
    y, sr = librosa.load(flac_file, sr=None)
    sf.write(output_path, y, sr)


def process_data(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_records_file = os.path.join(output_folder, 'processed_records.txt')
    processed_records = set()

    if os.path.exists(processed_records_file):
        with open(processed_records_file, 'r') as f:
            processed_records = set(f.read().splitlines())

    new_records = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.flac') and file not in processed_records:
                flac_path = os.path.join(root, file)
                wav_file_name = file.replace('.flac', '.wav')
                wav_output_path = os.path.join(output_folder, 'wavs', wav_file_name)

                convert_flac_to_wav(flac_path, wav_output_path)

                text_file = file.replace('.flac', '.TextGrid')
                text_path = os.path.join(root, 'TextGrid', text_file)
                text_path = text_path.replace("/wav", "")

                with open(text_path, 'r') as text_file:
                    speech_text = text_file.read().replace('\n', '|')

                new_records.append((wav_file_name, speech_text))
                processed_records.add(file)

    with open(processed_records_file, 'a') as f:
        for record in processed_records:
            f.write(record + '\n')

    with open(os.path.join(output_folder, 'metadata.csv'), 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')
        for record in new_records:
            csv_writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description='Process audio and text data')
    parser.add_argument('--input-path', required=True, help='/data/dataset/train_L')
    parser.add_argument('--output-path', required=True, help='/data/dataset/train_L/train_data')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    process_data(input_path, output_path)


if __name__ == "__main__":
    main()
