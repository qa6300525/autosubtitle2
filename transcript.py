#!/usr/bin/env python
# coding: utf-8
import os
import time

import whisper
import utils
import ffmpeg
import shlex


@utils.logit_time
def add_subtitles_to_video_cmd(video_path, subtitle_paths, output_path):
    if os.path.isfile(output_path):
        print("output file exists")
        return
    video_path = shlex.quote(video_path)
    subtitle_paths = [shlex.quote(subtitle_path) for subtitle_path in subtitle_paths]
    output_path = shlex.quote(output_path)

    cmd_subtitles = ""
    if len(subtitle_paths) == 1:  # 一个字幕
        cmd_subtitles += f'-vf subtitles="{subtitle_paths[0]}:force_style=\'Fontsize=18\'" '
    elif len(subtitle_paths) == 2:  # 两个字幕
        cmd_subtitles += f"""-vf \"subtitles={subtitle_paths[0]}:force_style='Fontsize=18,PrimaryColour=&H00FFFFFF,
        Alignment=2',subtitles={subtitle_paths[1]}:force_style='Fontsize=24,PrimaryColour=&H0000FFFF,MarginV=30,
        Alignment=2'\" """
    cmd_line = f'ffmpeg -i {video_path} {cmd_subtitles} {output_path}'
    print(cmd_line)
    import subprocess
    retcode = subprocess.call(cmd_line, shell=True)
    print(retcode)


@utils.logit_time
def add_subtitles(input_video, subtitle_file, zh_subtitle_file, output_video, overwrite=True):
    """
    添加字幕到视频文件，并避免字幕重叠。

    参数:
        input_video (str): 输入视频文件路径。
        subtitle_file (str): 输入字幕文件路径 (英文)。
        zh_subtitle_file (str): 输入字幕文件路径 (中文)。
        output_video (str): 输出视频文件路径。

    返回:
        None
    """

    # 设置字幕样式
    en_subtitle_style = "Fontsize=18,Alignment=2,PrimaryColour=&H00FFFFFF"
    zh_subtitle_style = "Fontsize=24,MarginV=30,Alignment=2,PrimaryColour=&H0000FFFF"

    # 使用 ffmpeg-python 添加字幕
    stream = ffmpeg.input(input_video)

    if os.path.isfile(subtitle_file):
        stream = stream.filter("subtitles", subtitle_file, force_style=en_subtitle_style)

    if os.path.isfile(zh_subtitle_file):
        stream = stream.filter("subtitles", zh_subtitle_file, force_style=zh_subtitle_style)
    if os.path.exists(output_video):
        if overwrite:
            os.remove(output_video)
        else:
            raise FileExistsError(f"Output file '{output_video}' already exists.")
    (
        stream
            .output(output_video, c="a", vcodec="libx264", acodec="copy", crf="23", format="mp4")
            .global_args("-movflags", "faststart")
            .run()
    )


@utils.logit_time
def extract_subtitle(path, video_name, srt_name, model_size, language):
    """
    Extracts subtitles from a video file using a pre-trained whisper model and writes them to an SRT file.

    :param path: The path of the directory containing the video file.
    :param video_name: The name of the video file.
    :param model_size: The size of the pre-trained whisper model to use for transcription.
    :param language: The language of the audio in the video file.
    :return: None.
    """
    srt_path = path + srt_name
    if os.path.exists(srt_path):
        print(f"Subtitle file '{srt_path}' already exists.")
        return

    print('Loading model...')
    filepath = os.path.join(path, video_name)
    model = whisper.load_model(model_size)

    dirname = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)

    print('Transcribe in progress...')
    result_origin = model.transcribe(audio=f'{filepath}', language=language, verbose=False)
    result_origin.get('language', language)

    print('Done')
    from whisper.utils import WriteSRT

    with open(srt_path, "w", encoding="utf-8") as srt:
        writer = WriteSRT(dirname)
        options = {"max_line_width": 40, "max_line_count": 40, "highlight_words": True}
        writer.write_result(result_origin, srt, options=options)



if __name__ == "__main__":
    # add_subtitles(
    #     "./data/Nerfs.mp4",
    #     "./data/output_video_4.mp4",
    #     "/Users/chengmaoyu/code/test/chatgpt/autosubtitle/data/Nerfs.srt",
    #     "/Users/chengmaoyu/code/test/chatgpt/autosubtitle/data/zh_Nerfs.srt"
    # )
    pass
