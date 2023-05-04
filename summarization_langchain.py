#!/usr/bin/env python
# coding: utf-8
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from translate import translate_with_chatgpt

import utils
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


# summarize with langchain
def summarize_with_langchain(path, filename):
    # Load your documents
    with open(path + filename, encoding="utf-8") as f:
        text = f.read()
    new_text = utils.extract_text_from_subtitle(text)
    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

    # Split your docs into texts
    texts = text_splitter.split_text(new_text)
    from langchain.docstore.document import Document

    docs = [Document(page_content=t) for t in texts]

    # There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"),
                 openai_api_base='https://api.openai-proxy.com/v1')
    user_prompt = """
    Task: 
    1. Generate a concise summary of the Text with a focus on at most 4 main topics. 
    every topic at most 20 words and output format like 1. ... , 2. ..., 3. ...:
    2. Generate an attractive title based on topic for your YouTube title.
    Text: ```{text}```
    """
    print(user_prompt)
    PROMPT = PromptTemplate(template=user_prompt, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True,
                                 combine_prompt=PROMPT)
    ans_text = chain.run(docs)
    # 输出结果
    return ans_text


if __name__ == '__main__':
    o = summarize_with_langchain('./data/', 'Nerfs.srt')
    t = translate_with_chatgpt(o, 'zh')
    print(o)
    print(t)
