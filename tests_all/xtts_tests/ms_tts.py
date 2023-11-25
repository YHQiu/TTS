import json

from azure.cognitiveservices.speech import SpeechConfig, AudioDataStream
import azure.cognitiveservices.speech as speechsdk
import config
import os

speech_config_zh_CN_XiaochengNeural = speechsdk.SpeechConfig(subscription=config.MS_SPEECH_KEY,
                                                             region=config.MS_SPEECH_REGION)
speech_config_zh_CN_XiaochengNeural.speech_synthesis_language = "zh-CN"
speech_config_zh_CN_XiaochengNeural.speech_synthesis_voice_name = "zh-CN-XiaochenNeural"

"""
https://speech.microsoft.com/portal/7050ff6233ea4939b5d4de5b0124af49/voicegallery
发音人：晓秋，女，zh-CN-XiaoqiuNeural
发音人：小陈，女，zh-CN-XiaochenNeural
"""


def get_speech_config(languange_name, voice_name):
    speech_config = speechsdk.SpeechConfig(subscription=config.MS_SPEECH_KEY, region=config.MS_SPEECH_REGION)
    speech_config.speech_synthesis_language = languange_name
    speech_config.speech_synthesis_voice_name = voice_name
    return speech_config


audio_config = None  # Use the default audio_tools output configuration


async def ms_synthesize_speech(text: str, file_path: str, file_name: str, speech_name: str = "xiaochen"):
    # 微软Audio 语音合成服务
        if speech_name == "xiaochen":
            speech_config = speech_config_zh_CN_XiaochengNeural
        else:
            speech_config = speech_config_zh_CN_XiaochengNeural

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,
                                                         audio_config=audio_config)

        result = speech_synthesizer.speak_text_async(text)

        stream = AudioDataStream(result.get())

        # Create the directory if it doesn't exist
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        file_path_name = os.path.join(file_path, file_name)

        # Generate a unique filename for the audio_tools file
        stream.save_to_wav_file(file_path_name)

        return

import json
import csv

def jsonl_to_csv(jsonl_file, output_csv):
    with open(jsonl_file, 'r', encoding='utf-8') as json_file, open(output_csv, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')

        lines = json_file.readlines()

        for index, line in enumerate(lines, start=1):
            json_obj = json.loads(line)
            text = json_obj['text'].replace('\n', ' ')  # Replace newline characters with spaces
            file_name = f"20231125_M_R001S01C01_01_{index:02}"

            writer.writerow([f"{file_name}", text])

if __name__ == "__main__":
    jsonl_to_csv('test.jsonl', 'metadata.csv')
    # async def process_jsonl_file(file_path: str):
    #     with open(file_path, 'r', encoding='utf-8') as json_file:
    #         lines = json_file.readlines()
    #
    #         for index, line in enumerate(lines, start=1):
    #             json_obj = json.loads(line)
    #             text = json_obj['text']
    #             # Generate the file name dynamically
    #             file_name = f"20231125_M_R001S01C01_01_{index:02}.wav"
    #
    #             # Perform speech synthesis for each text
    #             await ms_synthesize_speech(text, "train_100_data", file_name)
    #
    # import asyncio
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(process_jsonl_file('test.jsonl'))




