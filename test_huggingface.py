#!/usr/bin/env python
# coding: utf-8


from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import utils
from peft import LoraConfig
from build_dataset import buid_instruction_dataset, DataCollatorForSupervisedDataset



def load_model(text_list, src_lang, tgt_lang):
    global model
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model.save_pretrained("./data/model/m2m100_418M_m")

    global tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    tokenizer.save_pretrained("./data/model/m2m100_418M_t")

if __name__ == '__main__':
    load_model([""], "en", "zh")
