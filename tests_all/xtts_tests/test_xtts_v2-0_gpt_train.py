import argparse
import json
import os
from datetime import datetime

import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

def train_model(train_config):

    XTTS_CHECKPOINT = train_config.get("XTTS_CHECKPOINT")
    LANGUAGE_BASE = train_config.get("LANGUAGE_BASE")
    # Training Parameters
    EPOCHS = train_config.get("EPOCHS")
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = train_config.get("OPTIMIZER_WD_ONLY_ON_WEIGHTS") # for multi-gpu training please make it False
    START_WITH_EVAL = train_config.get("START_WITH_EVAL") # if True it will star with evaluation
    BATCH_SIZE = train_config.get("BATCH_SIZE")
    BATCH_GROUP_SIZE = train_config.get("BATCH_GROUP_SIZE")
    GRAD_ACUMM_STEPS = train_config.get("GRAD_ACUMM_STEPS")

    # DATA SET
    DATASET_PATH = train_config.get("DATASET_PATH")
    DATASET_WAV_PATH = train_config.get("DATASET_WAV_PATH")
    TRAIN_CSV = train_config.get("TRAIN_CSV")
    TEST_CSV = train_config.get("TEST_CSV")

    # DATA FORMATTER
    DATASET_FORMATTER = train_config.get("DATASET_FORMATTER")
    DATASET_NAME = train_config.get("DATASET_NAME")

    # Training sentences generations
    SPEAKER_REFERENCE = []
    added_speakers = set()  # 用于记录已添加的发音人

    # 输出文件路径
    # 获取当前日期并以特定格式保存
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更新路径中的日期部分
    OUT_PATH = os.path.join(train_config.get("OUT_PATH"), "train_outputs", current_date)
    os.makedirs(OUT_PATH, exist_ok=True)

    # 读取测试数据的发音人信息
    if train_config.get("SPEAKER_REFERENCE") is None:
        with open(TEST_CSV, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split('|')
                file_info = parts[0].split('_')
                speaker_id = '_'.join(file_info[:3])  # 获取前三段信息，用于唯一标识发音人
                if speaker_id not in added_speakers:
                    wav_file_name = f"{parts[0]}.wav"
                    speaker_wav_path = os.path.join(DATASET_WAV_PATH, wav_file_name)
                    SPEAKER_REFERENCE.append(speaker_wav_path)
                    added_speakers.add(speaker_id)

                full_wav_file_name = '_'.join(file_info[:4])  # 获取完整的文件名
                full_wav_file_path = os.path.join(DATASET_WAV_PATH, f"{full_wav_file_name}.wav")
                SPEAKER_REFERENCE.append(full_wav_file_path)
    else:
        SPEAKER_REFERENCE = train_config.get("SPEAKER_REFERENCE")

    print(SPEAKER_REFERENCE[0])

    config_dataset = BaseDatasetConfig(
        formatter=DATASET_FORMATTER,
        dataset_name=DATASET_NAME,
        path=DATASET_PATH,
        meta_file_train=TRAIN_CSV,
        meta_file_val=TEST_CSV,
        language=LANGUAGE_BASE,
    )

    DATASETS_CONFIG_LIST = [config_dataset]

    # Logging parameters
    RUN_NAME = "GPT_XTTS_LJSpeech_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Create DVAE checkpoint and mel_norms on test time
    # DVAE parameters: For the training we need the dvae to extract the dvae tokens, given that you must provide the paths for this model
    DVAE_CHECKPOINT = os.path.join(OUT_PATH, "dvae.pth")  # DVAE checkpoint
    # Mel spectrogram norms, required for dvae mel spectrogram extraction
    MEL_NORM_FILE = os.path.join(OUT_PATH, "mel_stats.pth")
    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    torch.save(dvae.state_dict(), DVAE_CHECKPOINT)
    mel_stats = torch.ones(80)
    torch.save(mel_stats, MEL_NORM_FILE)


    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = f"vocab.json"  # vocab.json file

    LANGUAGE = config_dataset.language

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length = 200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # 读取模型配置文件
    # with open("config.json", 'r') as model_config_file:
    #     model_args = json.load(model_config_file)

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        epochs=EPOCHS,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=BATCH_GROUP_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "看起来报错信息显示有 3 个活动的 GPU，并建议使用 CUDA_VISIBLE_DEVICES 来定义目标 GPU。您设置了 CUDA_VISIBLE_DEVICES 环境变量，但可能还需要确认是否正确设置了只使用一个 GPU。",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    trainer.fit()

if __name__ == "__main__":

    # 设置要使用的GPU
    # 获取CUDA_VISIBLE_DEVICES环境变量的值
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        # 获取当前CUDA版本
        cuda_version = torch.version.cuda
        print(f"当前使用的CUDA版本为: {cuda_version}")
    else:
        print("CUDA 不可用")

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # 默认为'0'

    # 将获取的值以逗号分隔拆分成一个GPU索引的列表
    gpu_list = cuda_visible_devices.split(',')

    # 循环遍历列表，设置每个GPU并启动你的代码
    for gpu_index in gpu_list:
        try:
            gpu_idx = int(gpu_index)
            torch.cuda.set_device(gpu_idx)
            print(f"Use GPU Index {gpu_idx}")
        except ValueError:
            print(f"Invalid GPU index: {gpu_index}")

    parser = argparse.ArgumentParser(description='Train model with specified configuration')
    parser.add_argument('--train-config-path', required=False, default="train_config.json", help='Path to train_config.json')
    args = parser.parse_args()

    config_path = args.train_config_path

    # 从 train_config.json 中读取配置
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # 将读取的配置传递给 train_model 函数
    train_model(config)
