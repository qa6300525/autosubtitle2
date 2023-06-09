#!/usr/bin/env python
# coding: utf-8


import ffmpeg
from PIL import Image, ImageDraw, ImageFont
import os

import utils
import imgkit
from jinja2 import Template
import markdown
from io import BytesIO


def create_text_image_v1(text: str, image_name: str,
                         image_size=None) -> None:
    if image_size is None:
        image_size = (800, 400)
    width, height = image_size

    markdown_text = text

    html_template = '''
    <!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            font-size: 23px;
            line-height: 1.8;
            text-align: justify;
            # max-width: 1600px;
            display: flex;
            word-wrap: break-word;
            margin: auto;
            padding: 50px;
            background-color: #f0f0f0;
        }

        div {
            width: 1600px;
            height: 900px;
            overflow-y: auto;
            font-size: 1.25em;
            line-height: 1.5em;
            background-color: #ffffff;
            padding: 2%;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div>
        {{ content }}
    </div>
</body>
</html>

    '''
    html_content = markdown.markdown(markdown_text)
    template = Template(html_template)
    html_output = template.render(content=html_content)

    options = {
        "format": "png",
        "quality": 100,
        "crop-h": height,
        "crop-w": width,
        "encoding": "UTF-8",
    }

    img_bytes = imgkit.from_string(html_output, False, options=options)

    with BytesIO(img_bytes) as b:
        img = Image.open(b)
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        target_aspect_ratio = 16 / 9
        if aspect_ratio < target_aspect_ratio:
            new_width = min(width, original_width)
            new_height = int(new_width / target_aspect_ratio)
        else:
            new_height = min(height, original_height)
            new_width = int(new_height * target_aspect_ratio)

        img_16_9 = img.resize((new_width, new_height), 1)
        img_16_9.save(image_name)


def create_text_image(text: str, image_name: str, font_path: str, font_size: int,
                      image_size=None) -> None:
    if image_size is None:
        image_size = (800, 400)
    width, height = image_size

    # 创建一个白色背景的图片
    image = Image.new('RGB', (width, height), (255, 255, 255))

    # 选择字体和大小
    font = ImageFont.truetype(font_path, font_size)
    # 计算文本大小

    # 计算文本位置
    x = image.width / 10
    y = image.height / 6
    print(x, y)
    # 计算文本大小以覆盖图片的80％

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 处理中文字符串
    text = utils.process_chinese_strings(text)

    # 绘制文本
    draw.multiline_text((x, y), text, font=font, fill=(0, 0, 0), align='left')
    # 保存图片
    image.save(image_name)


def add_image_to_video(input_video: str, output_video: str, image_name: str, duration: int,
                       overwrite=True) -> None:
    # 获取输入视频的信息（宽度、高度、帧率）
    probe = ffmpeg.probe(input_video)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width, height = video_info['width'], video_info['height']
    framerate = video_info['r_frame_rate']

    # 将图片与视频合成
    image_stream = (
        ffmpeg
            .input(image_name, loop=1, framerate=framerate, t=duration)
            .filter('scale', width, height)
    )

    video_stream = ffmpeg.input(input_video)

    # 获取音频流
    audio_stream = video_stream.audio
    delayed_audio_stream = audio_stream.filter('adelay', f"{duration * 1000}|{duration * 1000}")

    # 合并视频流
    merged_video = ffmpeg.concat(image_stream, video_stream, v=1, a=0)

    # 合并音频和视频流
    output_stream = ffmpeg.output(
        merged_video,
        delayed_audio_stream,
        output_video,
    ).overwrite_output()

    if os.path.exists(output_video):
        if overwrite:
            os.remove(output_video)
        else:
            raise FileExistsError(f"Output file '{output_video}' already exists.")
    # 运行ffmpeg命令
    output_stream.run()


def main():
    text = '''1. Nerf是一种使用2D图像创建3D模型的3D重建技术。
2. ControlNet是一种为约束优化添加额外条件的稳定扩散方法。
3. Dreamboot 3D paper是一种将个性化AI与Nerf相结合的三步过程。
4. 文本嵌入和诱饵以及控制元素被用来使图像更加一致。
5. Marching cube算法和Misa优化器被用来从3D数据创建网格并生成未见过的视角。'''

    create_text_image(text=text, image_name='text_image.png',
                      font_path='arial.ttf', font_size=24, image_size=(800, 400))
    add_image_to_video(input_video='test.mp4', output_video='output_test.mp4',
                       image_name='text_image.png', duration=5)


if __name__ == '__main__':
    with open("./data/AutoGPT Tutorial - More Exciting Than ChatGPT [FeIIaJUN-4A]_summary.txt", "r") as f:
        text = f.read()
    # create_text_image(text=text, image_name='./data/text_image.png',
    #                   font_path='/System/Library/Fonts/STHeiti Light.ttc'
    #                   , font_size=50, image_size=(1920, 1080))

    create_text_image_v1(text=text, image_name='./data/text_image.png',
                         font_path=f'{utils.get_cur_dir()}/STHeiti Light.ttc'
                         , font_size=50, image_size=(1920, 1080))
    # add_image_to_video(input_video='./data/Nerfs_subtitled.mp4', output_video='./data/output_test.mp4',
    #                    image_name='./data/text_image.png', duration=5)
