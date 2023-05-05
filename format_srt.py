#!/usr/bin/env python
# coding: utf-8
import re
from typing import Tuple, List


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


def split_subtitle_line_zh(line: str, max_length: int) -> List[str]:
    lines = []
    current_line = ""
    current_length = 0

    for char in line:
        if is_chinese(char):
            current_length += 2
        else:
            current_length += 1

        if current_length <= max_length:
            current_line += char
        else:
            lines.append(current_line)
            current_line = char
            current_length = 2 if is_chinese(char) else 1

    if current_line:
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
    output = []
    index = 1
    for line in content.split("\n"):
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line):
            timestamp = line
        elif line.isdigit():
            pass
        elif line.strip():
            if is_chinese(line):
                split_lines = split_subtitle_line_zh(line, max_length)
            else:
                split_lines = split_subtitle_line_en(line, max_length)
            if len(split_lines) > 1:
                for i, new_line in enumerate(split_lines):
                    new_start, new_end = split_timestamp(timestamp, i, len(split_lines))
                    output.append(str(index))
                    output.append(f"{new_start} --> {new_end}")
                    output.append(new_line)
                    output.append('\n')
                    index += 1
            else:
                output.append(str(index))
                output.append(timestamp)
                output.extend(split_lines)
                index += 1

    return "\n".join(output)


def split_subtitles(input_file: str, output_file: str, max_length: int) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    processed_content = process_subtitle_file(content, max_length)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(processed_content)


if __name__ == "__main__":
    input_file = "./data/The LangChain Cookbook Part 2 - Beginner Guide To 9 Use Cases.srt"
    output_file = "output.srt"
    max_length = 42

    split_subtitles(input_file, output_file, max_length)
