from flask import Flask, redirect, render_template, request
from nmt_model import Hypothesis, NMT
from run import beam_search, compute_corpus_level_bleu_score
from utils import read_corpus, batch_iter, separate_sentences, read_input_sentences, clean_sentences
import re

app = Flask(__name__)

@app.route('/')
def main():
    return redirect('/translate')


def translate_en_to_vi(english_sentences):

    test_data_src = read_input_sentences(english_sentences)
    model = NMT.load('model_en_vi_2.bin')
    hypotheses = beam_search(model, test_data_src, beam_size=10, max_decoding_time_step=150)

    source_sentences, translated_sentences = [], []
    for t, h in zip(test_data_src, hypotheses):
        source_sentence = ' '.join(t)
        source_sentences.append(source_sentence)
        translated_sentence = ' '.join(h[0].value)
        clean_translated_sentence = clean_sentences(translated_sentence, type="output_sentences")
        translated_sentences.append(clean_translated_sentence)

    # squeeze_list = [" ".join(sent) for sent in translated_sentences]
    list_to_str = "\n".join(translated_sentences)

    return list_to_str


@app.route('/translate', methods=['GET', 'POST'])
def translate():
    english_sentences = request.form.get('english')
    # if input is valid (not empty)
    if english_sentences:
        english_sentences = english_sentences
    else:
        default_example = "This system, which is created by Kevin, can translate English to Vietnamese.\
        Please test it out yourself."
        english_sentences = default_example

    separated_english_sentences = separate_sentences(english_sentences)
    clean_english_sentences = clean_sentences(separated_english_sentences)
    translation = translate_en_to_vi(clean_english_sentences)

    return render_template("index2.html", english=clean_english_sentences, vietnamese=translation)


if __name__ == '__main__':
    app.run(port=5000, debug=False)