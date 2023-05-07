#!/usr/bin/env python
# coding: utf-8


from translate import translate_gpt, translate, user_open_ai


def test_translate():
    file = "./data/temp.srt"
    input_language, output, output_language, chunk_size = "English", "./data/nerfs.srt", "Chinese", 1000
    chunks = translate_gpt(file, input_language, output, output_language, chunk_size)
    print(len(chunks))
    for i, chunk in enumerate(chunks):
        translate(chunk, "English", "Chinese")
        break


def test_translate_gpt():
    path = "./data/"
    input_filename = "Nerfs.srt"
    input_language, output_language, chunk_size = "en", "zh", 1000
    output = f"{output_language}_{input_filename}"
    translate_gpt(path, input_filename, input_language, output, output_language, chunk_size)


def test_user_open_ai():
    a = user_open_ai("just say this is a test")
    print(a)


if __name__ == "__main__":
    # test_translate()
    # test_translate_gpt()
    # test_user_open_ai()
    import imgkit
    import markdown2
    markdown_text = '''
## 总结

该视频介绍了一个名为**Humata**的AI，可以用来快速概括长PDF文档。它还可以回答关于文档的具体问题，并提供编写文档和将文本文件编译为PDF文件的指导。它比*ChatGDP*更强大，可用于学校工作。

### 主题

1. 介绍Humata AI
2. 注册Humata AI
3. 上传PDF文档
4. 概括文档
5. Humata AI的好处

**标签**：AI，Humata，PDF，概括，学校工作
'''
    html_text = markdown2.markdown(markdown_text)

    options = {
        'format': 'png',
        'encoding': "UTF-8",
        'zoom': 1.3,
        'viewport-size': '1920x1080'
    }

    imgkit.from_string(html_text, 'output.png', options=options)
