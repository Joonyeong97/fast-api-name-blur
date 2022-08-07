import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

model_name = './models/nameblur'

class NerModel:
    def __init__(self, model_name, max_len):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        self.max_len = max_len

    def _get_ner_tag(self, text):
        return self.nlp(text)

    def _subtract_index(self, index: list):
        return np.abs(index[0] - index[1])

    def _get_split_max_len_text(self, text):
        split_strings = []
        for index in range(0, len(text), self.max_len):
            split_strings.append(text[index: index + self.max_len])

        return split_strings

    def __call__(self, *args, **kwargs):

        raise ''

class NameBlurNer(NerModel):
    def __init__(self, model_name, max_len=65, blur_string='*'):
        super().__init__(model_name, max_len)
        self.blur_string = blur_string
        self.korea_regex = r"[ㄱ-ㅣ가-힣+]"

    def _blur_index(self, string, blur_str, index: list):
        start = index[0]
        sub_index = self._subtract_index(index)
        blur_str = blur_str * sub_index
        return string[:start] + blur_str + string[start + sub_index:]

    def _get_tag_blur_index(self, ner_tag):
        tag_list = []
        for tag_dict in ner_tag:
            if tag_dict['entity'] == 'PER-B' or tag_dict['entity'] == 'PER-I':

                if re.findall(self.korea_regex, tag_dict['word']):

                    index = [tag_dict['start'], tag_dict['end']]
                    tag_list.append(index)
        return tag_list

    def _blur_text(self, text, tag_list):
        blur_text = text

        for tag_index in tag_list:
            blur_text = self._blur_index(blur_text, self.blur_string, tag_index)
        return blur_text

    def _return_blur_text(self, text):
        ner_tag = self._get_ner_tag(text)
        tag_list = self._get_tag_blur_index(ner_tag)
        blur_text = self._blur_text(text, tag_list)

        return blur_text

    def __call__(self, text):
        input_text = ' '.join(text.split())
        if len(input_text) > self.max_len:
            split_strings = self._get_split_max_len_text(input_text)
            blur_texts = [self._return_blur_text(split_string) for split_string in split_strings]
            blur_text = ''.join(blur_texts)

        else:
            blur_text = self._return_blur_text(input_text)

        return blur_text
