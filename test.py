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
    test_translate_gpt()
    # test_user_open_ai()
