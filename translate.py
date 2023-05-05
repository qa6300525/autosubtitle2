#!/usr/bin/env python
# coding: utf-8
import openai
import argparse
import time
import tiktoken
import re
import os

from dotenv import load_dotenv, find_dotenv
import translate_huggingface
import utils

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = 'https://api.openai-proxy.com/v1'


# print(openai.api_key)


# check number of tokens of a message
def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = len(encoding.encode(message))
        return num_tokens
    else:
        raise NotImplementedError(f"""error.""")


# group messages together
def group_chunks(chunks, ntokens, max_len=1000):
    """
    Group very short chunks, to form approximately a page long chunks.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0

    # iterate over chunks, and group the short ones together
    for chunk, ntoken in zip(chunks, ntokens):
        # print(ntoken)
        # notken = num_tokens_from_messages(chunk)
        cur_tokens += ntoken + 2  # +2 for the newlines between chunks

        # if adding this chunk would exceed the max length, finalize the current batch and start a new one
        if ntoken + cur_tokens > max_len:
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = 0
        else:
            cur_batch += "\n\n" + chunk
            # cur_batch += chunk
    batches.append(cur_batch)
    return batches


def write_srt_file(filename, subtitles):
    with open(filename, 'w') as f:
        f.write(subtitles)
        # for subtitle in subtitles:
        #     f.write(subtitle)


def write_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)


def parse_srt_data(srt_data):
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n\n'
    matches = re.findall(pattern, srt_data, re.DOTALL)
    subtitles = []
    for match in matches:
        subtitle = {
            'number': int(match[0]),
            'start_time': match[1],
            'end_time': match[2],
            'text': match[3].strip()
        }
        subtitles.append(subtitle)
    return subtitles


def user_open_ai(content):
    import requests
    import json

    url = 'https://chengdd.top/api/openaiv4'
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0
    }
    headers = {'Content-type': 'application/json', 'Authorization': "Bearer " + openai.api_key}
    print(json.dumps(data))
    response = requests.post(url, data=json.dumps(data), headers=headers)

    # print(response.text)
    import json
    print(response.text)
    completion = json.loads(response.text)
    return completion['data']


def summarize_with_chatgpt(text, output_language):
    prompt_text = f"""
Task: input is a text, summarize the text with {output_language}, like 1. ... 2. ... 3. ... format.
Text: ```{text}``` 
"""
    return call_open_ai(prompt_text)


def translate(text, input_language, output_language):
    prompt_text = f"""
Task: Do several tasks, output as a json object with "Translate", "Summarize", "Translate summarize" as the key:
1. Translate:Translate the text from {input_language} to {output_language}. \
 For each chunk, the first row is a number and the \
second row is a text. Only translate the text part and keep the format.
2. Summarize:Summarize the text with English. 
3. Translate summarize:Translate the summarize from English to {output_language}. 

Text: ```{text}``` """
    # print(prompt_text)
    import json
    ans = json.loads(call_open_ai(prompt_text))
    return ans


def translate_with_chatgpt(text, output_language):
    prompt_text = f"""
Task: input is a text, translate the text to {output_language},
Only translate the text part and keep the format. 
Note: 
1. Unable to understand or professional vocabulary is kept, do not need to translate.
2. The translation is clear and concise, no more than 20 words.
Text: ```{text}``` """
    return call_open_ai(prompt_text)


def generate_text(prompt, model, length=50, temperature=0.5):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=length,
        temperature=temperature,
    )
    return response.choices[0].text.strip()


def call_open_ai(prompt_text):
    t_text = ""
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        # completion = user_open_ai(prompt_text)
        t_text = (
            completion["choices"][0]
                .get("message")
                .get("content")
                .encode("utf8")
                .decode()
        )
        # format the translated text, the original text is eg: "\n\n['\\n柠檬\\n\\n', '梶井基次郎']", we need the
        # element in the list, not the \n \n

        # openai has a time limit for api  Limit: 20 / min
    except Exception as e:
        print(str(e), "will sleep 60 seconds")
    return t_text


@utils.logit_time
def translate_gpt(path, input, input_language, output, output_language, chunk_size):
    file = path + input
    file_out = path + output
    if os.path.exists(file_out):
        print("file exists, skip")
        return

    with open(file, "r") as f:
        text = f.read()

    translated_subtitle, summarize_with_english, summarize_with_target_language = \
        translate_subtitle(text, input_language, output_language, chunk_size)
    write_srt_file(file_out, translated_subtitle)

    return file_out


def split_text(text_list, max_chunk_length):
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for text in text_list:
        text_length = len(text)
        if current_chunk_length + text_length <= max_chunk_length:
            current_chunk.append(text)
            current_chunk_length += text_length
        else:
            chunks.append(current_chunk)
            current_chunk = [text]
            current_chunk_length = text_length
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def translate_text(text_list, input_language, output_language, chunk_size):
    text_chunks = split_text(text_list, chunk_size)
    translated_chunks = []
    english_summarize = []
    output_language_summarize = []
    for i, chunk in enumerate(text_chunks):
        print(str(i + 1) + " / " + str(len(text_chunks)))
        ans = translate(chunk, input_language, output_language)

        translated_chunks.extend(ans.get('Translate'))
        english_summarize.append(ans.get('Summarize'))
        output_language_summarize.append(ans.get('Translate summarize'))
    # return translated_chunks, english_summarize, chinese_summarize
    # join the chunks together

    return translated_chunks, english_summarize, output_language_summarize


def translate_text_local(text_list, input_language, output_language, chunk_size):
    english_summarize = []
    output_language_summarize = []
    translated_chunks = translate_huggingface.translate(text_list, input_language, output_language)

    return translated_chunks, english_summarize, output_language_summarize


def restore_subtitle_format(translated_text, original_subtitle):
    subtitle_lines = original_subtitle.split('\n')
    new_subtitle = ""
    idx = 0
    line_count = len(subtitle_lines)
    i = 0
    while i < line_count:
        line = subtitle_lines[i]
        if re.match(r'\d+', line) and i + 2 < line_count and re.match(
                r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', subtitle_lines[i + 1]):
            new_subtitle += line + "\n" + subtitle_lines[i + 1] + "\n" + translated_text[idx] + "\n"
            idx += 1
            i += 4
        else:
            new_subtitle += line + "\n"
            i += 1
    print(new_subtitle)
    return new_subtitle.strip()


def user_restore_subtitle_format(translated_text, original_subtitle):
    subtitle_lines = original_subtitle.split('\n')
    idx = 2
    i = 0
    line_count = len(subtitle_lines)
    t_count = len(translated_text)
    while idx < line_count and i < t_count:
        subtitle_lines[idx] = translated_text[i][0]
        i += 1
        idx += 4
    print(line_count, t_count, i, idx)
    new_subtitle = "\n".join(subtitle_lines)
    return new_subtitle


def translate_subtitle(subtitle, input_language, output_language, chunk_size):
    extracted_text = utils.extract_text_from_subtitle(subtitle)
    translated_chunks, english_summarize, output_language_summarize = \
        translate_text_local(extracted_text, input_language, output_language, chunk_size)
    summarize_with_english, summarize_with_target_language = "", ""
    translated_subtitle = user_restore_subtitle_format(translated_chunks, subtitle)
    return translated_subtitle, summarize_with_english, summarize_with_target_language


if __name__ == "__main__":
    t = """
1. Nerf is a 3D reconstruction technique that uses 2D images to create 3D models.
2. ControlNet is a way of adding extra conditionals to stable diffusion for constrained optimization.
3. Dreamboot 3D paper is a three-step process that combines personalized AI with a Nerf.
4. Textual embeddings and lures, as well as control met are used to make images more consistent.
5. Marching cube algorithms and Misa optimizers are used to create a mesh from 3D data and generate unseen views."""
    o = translate_with_chatgpt(t, 'zh')
    print(o)
