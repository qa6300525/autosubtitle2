from datetime import datetime


def logit_time(func):
    """
    打印函数运行起始时间、终止时间、运行时间的装饰器
    """

    def print_func_run_time(*args, **kwargs):
        """
        打印函数运行时间
        """
        time1 = datetime.now()
        print(time1, f'{func.__name__} start running.')
        res = func(*args, **kwargs)
        time2 = datetime.now()
        print(time2, f'{func.__name__} done.')
        print(f'{func.__name__} total use time:', time2 - time1)
        return res

    return print_func_run_time


def extract_text_from_subtitle(subtitle):
    import re
    pattern = re.compile(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+)')
    return pattern.findall(subtitle)


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


def process_chinese_strings(strings):
    strings = strings.replace('\n', '\n\n\n')
    strings_list = strings.split('\n')
    processed_list = []
    slice_index = 38
    for s in strings_list:
        if len(s) > slice_index and s[slice_index] != '；':
            s = s[:slice_index] + '\n' + s[slice_index:]
        processed_list.append(s)
    return "\n".join(processed_list)

if __name__ == '__main__':
    with open("./data/zh_Chat GPT for files (AI better then Chat GPT) [oWfEjvI7aes]_zh_summary.txt", "r") as f:
        text = f.read()
    a = process_chinese_strings(text)
    print(a)
