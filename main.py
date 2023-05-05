#!/usr/bin/env python
# coding: utf-8

import os

import utils
from transcript import add_subtitles, extract_subtitle, add_subtitles_to_video_cmd
from translate import translate_gpt, translate, user_open_ai, translate_with_chatgpt
from summarization_langchain import summarize_with_langchain
from text_image import create_text_image, add_image_to_video
from format_srt import split_subtitles


def add_summary_to_video(path, summary_name, text, video_name, output_video_name):
    if not os.path.exists(path + video_name):
        print("video not exists")
        return
    if not os.path.exists(path + summary_name):
        print("summary not exists")
        return
    if os.path.exists(path + output_video_name):
        print("output video exists")
        return
    image_name = path + summary_name.split(".")[0] + ".png"

    font_path = f'{utils.get_cur_dir()}/resources/STHeiti Light.ttc'

    create_text_image(text=text, image_name=image_name,
                      font_path=font_path
                      , font_size=50, image_size=(1920, 1080))
    add_image_to_video(input_video=path + video_name, output_video=path + output_video_name,
                       image_name=image_name, duration=3)


if __name__ == "__main__":
    path = "./data/"
    # 创建文件夹
    os.makedirs(path, exist_ok=True)
    video_name = "Nerfs.mp4"

    # 1. extract subtitle from video
    video_path = path + video_name
    srt_name = video_name.split(".")[0] + ".srt"
    extract_subtitle(path, video_name, srt_name, "base", "en")

    # 2. translate subtitle
    input_language, output_language, chunk_size = "en", "zh", 1000
    output_language_srt = srt_name.split(".")[0] + f"_{output_language}.srt"
    translate_gpt(path, srt_name, input_language, output_language_srt, output_language, chunk_size)

    # 3. summarize subtitle
    summary_filename = srt_name.split(".")[0] + "_summary.txt"
    if os.path.exists(path + summary_filename):
        print("summary file exists")
    else:
        ans_text = summarize_with_langchain(path, srt_name)
        with open(path + summary_filename, "w") as f:
            f.write(ans_text)

    # 4. translate summary
    summary_zh_filename = output_language_srt.split(".")[0] + "_summary.txt"
    if os.path.exists(path + summary_zh_filename):
        print("summary file exists")
    else:
        with open(path + summary_filename, "r") as f:
            ans_text = f.read()
        t_ans_text = translate_with_chatgpt(ans_text, "zh")
        with open(path + summary_zh_filename, "w") as f:
            f.write(t_ans_text)

    # 5. format split subtitle
    # 5.1 split subtitle en
    format_srt_name = srt_name.split(".")[0] + "_format.srt"
    split_subtitles(path + srt_name, path + format_srt_name, 45)

    # 5.2 split subtitle zh
    format_output_language_srt = output_language_srt.split(".")[0] + "_format.srt"
    split_subtitles(path + output_language_srt, path + format_output_language_srt, 45)

    # 6. add subtitle to video
    output_video_name = video_name.split(".")[0] + "_subtitled.mp4"
    output_video_path = path + output_video_name
    subtitle_paths = [path + format_srt_name, path + format_output_language_srt]
    add_subtitles_to_video_cmd(video_path, subtitle_paths, output_video_path)

    # 6. add summary to video
    final_video_name = output_video_name.split(".")[0] + "_final.mp4"
    with open(path + summary_zh_filename, "r") as f:
        t_ans_text = f.read()
    add_summary_to_video(path, summary_filename, t_ans_text, output_video_name, final_video_name)
