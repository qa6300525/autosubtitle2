#!/usr/bin/env python
# coding: utf-8

import os

from transcript import add_subtitles, extract_subtitle, add_subtitles_to_video_cmd
from translate import translate_gpt, translate, user_open_ai, translate_with_chatgpt
from summarization_langchain import summarize_with_langchain
from text_image import create_text_image, add_image_to_video
from format_srt import split_subtitles


def add_summary_to_video(path, summary_name, text, video_name, output_video_name):
    image_name = path + summary_name.split(".")[0] + ".png"
    create_text_image(text=text, image_name=image_name,
                      font_path='/System/Library/Fonts/STHeiti Light.ttc'
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
    output_language_srt = f"{srt_name}_{output_language}"
    translate_gpt(path, srt_name, input_language, output_language_srt, output_language, chunk_size)

    # 3. summarize subtitle
    ans_text = summarize_with_langchain(path, srt_name)
    summary_filename = srt_name.split(".")[0] + "_summary.txt"
    with open(path + summary_filename, "w") as f:
        f.write(ans_text)

    # 4. translate summary
    with open(path + summary_filename, "r") as f:
        ans_text = f.read()
    t_ans_text = translate_with_chatgpt(ans_text, "zh")
    summary_filename = output_language_srt.split(".")[0] + "_summary.txt"
    with open(path + summary_filename, "w") as f:
        f.write(t_ans_text)

    # 5. format split subtitle
    # 5.1 split subtitle en
    format_srt_name = srt_name.split(".")[0] + "_format.srt"
    split_subtitles(path + srt_name, path + format_srt_name, 45)

    # 5.2 split subtitle zh
    format_srt_name = output_language_srt.split(".")[0] + "_format.srt"
    split_subtitles(path + output_language_srt, path + format_srt_name, 45)

    # 6. add subtitle to video
    output_video_name = video_name.split(".")[0] + "_subtitled.mp4"
    output_video_path = path + output_video_name
    subtitle_paths = [path + srt_name, path + output_language_srt]
    add_subtitles_to_video_cmd(video_path, subtitle_paths, output_video_path)

    # 6. add summary to video
    final_video_name = output_video_name.split(".")[0] + "_final.mp4"
    add_summary_to_video(path, summary_filename, t_ans_text, output_video_name, final_video_name)