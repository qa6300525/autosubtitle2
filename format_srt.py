#!/usr/bin/env python
# coding: utf-8
import re
from typing import Tuple, List

text_splitter = None


def split_subtitle_line_en(line: str, max_length: int) -> List[str]:
    words = line.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += f" {word}" if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def is_chinese(char):
    return re.match(r"[\u4e00-\u9fff]", char) is not None


def split_subtitle_line_opt(line: str, max_length: int) -> List[str]:
    lines = []
    current_line = ""
    current_length = 0
    separators = ["\n\n", "\n", ".", ",", "?", "!", "、", "。"]

    def split_string(s, delimiters):
        pattern = '|'.join(map(re.escape, delimiters))
        return re.split(pattern, s)

    if is_chinese(line[0]):
        max_length = int(max_length / 2)
    temp_lines = split_string(line, separators)
    for temp_line in temp_lines:
        if len(temp_line) + current_length <= max_length:
            current_line += temp_line
            current_length += len(temp_line)
        else:
            if len(current_line.strip()) > 0:
                lines.append(current_line)
            current_line = temp_line
            current_length = len(temp_line)
    current_line = current_line.strip()
    if len(current_line) > 0:
        lines.append(current_line)

    return lines


def split_timestamp(timestamp: str, index: int, length: int) -> Tuple[str, str]:
    start, end = timestamp.split(" --> ")
    start_ms = time_str_to_ms(start)
    end_ms = time_str_to_ms(end)
    interval_ms = end_ms - start_ms

    new_start_ms = start_ms + int(interval_ms / length * index) + 1
    new_end_ms = new_start_ms + int(interval_ms / length)

    new_start = ms_to_time_str(new_start_ms)
    new_end = ms_to_time_str(new_end_ms)

    return new_start, new_end


def split_timestamp_v2(timestamp: str, last_end: str,
                       total_length: int, cur_length: int) -> Tuple[str, str]:
    # 根据字幕长度分配时间
    start, end = timestamp.split(" --> ")
    start_ms = time_str_to_ms(start)
    end_ms = time_str_to_ms(end)
    interval_ms = end_ms - start_ms
    # 计算每个字符的展示时间 = 总时间 / 总长度
    interval_ms_per_char = int(interval_ms / total_length)

    # 计算当前字幕的开始时间 = 上一个字幕的结束时间
    if len(last_end) == 0:
        new_start = start
        new_start_ms = start_ms
    else:
        last_end_ms = time_str_to_ms(last_end)
        new_start_ms = last_end_ms + 1
        new_start = ms_to_time_str(new_start_ms)

    # 计算当前字幕的结束时间 = 开始时间 + 字幕长度 * 每个字符的展示时间
    new_end_ms = new_start_ms + cur_length * interval_ms_per_char

    # 如果结束时间超过了原字幕的结束时间，则将结束时间设置为原字幕的结束时间
    if new_end_ms > end_ms:
        new_end_ms = end_ms
    new_end = ms_to_time_str(new_end_ms)

    return new_start, new_end


def time_str_to_ms(time_str: str) -> int:
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    total_ms = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    return total_ms


def ms_to_time_str(ms: int) -> str:
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def process_subtitle_file(content: str, max_length: int) -> str:
    global timestamp
    output = []
    index = 1
    for line in content.split("\n"):
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line):
            timestamp = line
        elif line.isdigit():
            pass
        elif line.strip():
            # 统计总的长度
            total_length = len(line.encode("utf-8"))
            # 计算每个字符的展示时间

            if is_chinese(line):
                split_lines = split_subtitle_line_opt(line, max_length)
            else:
                split_lines = split_subtitle_line_en(line, max_length)
            if len(split_lines) > 1:
                last_end = ""
                for i, new_line in enumerate(split_lines):
                    # new_start, new_end = split_timestamp(timestamp, i, len(split_lines))
                    new_start, new_end = split_timestamp_v2(timestamp, last_end,
                                                            total_length, len(new_line.encode("utf-8")))
                    last_end = new_end
                    output.append(str(index))
                    output.append(f"{new_start} --> {new_end}")
                    output.append(new_line)
                    output.append('\n')
                    index += 1
            else:
                output.append(str(index))
                output.append(timestamp)
                output.extend(split_lines)
                output.append('\n')
                index += 1
    final_text = "\n".join(output)
    final_text = final_text.replace("\n\n\n", "\n\n")
    return final_text


def split_subtitles(input_file: str, output_file: str, max_length: int) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    processed_content = process_subtitle_file(content, max_length)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed_content)


if __name__ == "__main__":
    input_file = "./data/Build Your Own Auto-GPT Apps with LangChain (Python Tutorial) [NYSWn1ipbgg].txt"
    output_file = "output.srt"
    max_length = 52

    split_subtitles(input_file, output_file, max_length)
