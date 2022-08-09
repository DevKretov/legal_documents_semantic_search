import streamlit as st
import pandas as pd
import base64
#from PyPDF2 import PdfFileWriter, PdfFileReader
#from pdf2image import convert_from_path
import io
import unicodedata
from typing import List


import logging
logger = logging.getLogger('stanza').setLevel(logging.WARNING)

import fitz

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_sentence_transformer_model(model_name='all-MiniLM-L12-v2'):
    sentence_transformer = SentenceTransformer(model_name)
    return sentence_transformer


sentence_transformer = load_sentence_transformer_model()


@st.cache(allow_output_mutation=True)
def get_document_text_sentence_by_sentence_embedding(model:SentenceTransformer, sentences:List[str]):
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True,
                                             convert_to_tensor=True)
    return embeddings


def find_top_n_sentences_to_prompt(model:SentenceTransformer, embeddings, sentences:List[str], prompt_sentence, n):
   # prompt_sentence = 'the size of compensation'
    # prompt_sentence = 'scope of work'

    prompt_sentence_embedding = model.encode([prompt_sentence], convert_to_tensor=True)

    # cosine_scores = util.pairwise_dot_score(prompt_sentence_embedding, embeddings)
    cosine_scores = util.pairwise_cos_sim(prompt_sentence_embedding, embeddings)
    sentences_rankings = list(zip(sentences, cosine_scores.tolist()))
    sentences_rankings = sorted(sentences_rankings, key=lambda a: a[1], reverse=True)

    PRINT_TOP_N = n
    sentence_i = 0

    best_sentences = ''
    for sentence, ranking in sentences_rankings:
        print(f'\nRank: {ranking:3.5f}\n{sentence}')

        best_sentences += f'\nRank: {ranking:3.5f}\n{sentence}\n'
        sentence_i += 1
        if sentence_i == PRINT_TOP_N: break

    return best_sentences

def stanza_sent_tokenize(_text):
    import stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    doc = nlp(_text)

    sentences = []
    for sentence in doc.sentences:
        sentences.append(sentence.text)

    return sentences

def extract_sentences_from_pdf(pdf_bytes):
    texts = []

    with fitz.Document(stream=io.BytesIO(pdf_bytes)) as doc:
        for page in doc:
            text = page.get_text("blocks")
            texts.append(text)

    text_blocks = [list(map(lambda _entry: ' '.join(_entry[-3].split('\n')), text)) for text in texts]
    text_blocks = [list(map(lambda _text: unicodedata.normalize('NFKD', _text), text)) for text in text_blocks]

    text_blocks_all = []
    for _block in text_blocks:
        text_blocks_all.extend(_block)

    all_sentences = []

    for _block in text_blocks_all:
        all_sentences.extend(stanza_sent_tokenize(_block))

    return all_sentences


def extract_text_from_pdf(pdf_bytes):
    with fitz.Document(stream=io.BytesIO(pdf_bytes)) as doc:
        for page in doc:
            text = page.get_text("blocks")
            break

    text_blocks = list(map(lambda _entry: ' '.join(_entry[-3].split('\n')), text))
    return text_blocks

file_pdf = st.file_uploader("Upload a PDF file", type=(["pdf"]))


if file_pdf:

    input_bytes = file_pdf.getvalue()

  #  inputpdf = PdfFileReader(io.BytesIO(input_bytes))

    sentences = extract_sentences_from_pdf(input_bytes)
    embeddings = get_document_text_sentence_by_sentence_embedding(sentence_transformer, sentences)


    base64_pdf = base64.b64encode(io.BytesIO(input_bytes).read()).decode('utf-8')
    pdf_display = f'<p align="center"><iframe src="data:application/pdf;base64,{base64_pdf}#zoom=130" width="50%" height="600" type="application/pdf"></iframe></p>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    form = col1.form(key='my-form')
    prompt_sentence = form.text_input('Enter prompt sentence')
    submit = form.form_submit_button('Find')

    st.write('Press submit to have your name printed below')

    if submit:
        best_sentences = find_top_n_sentences_to_prompt(sentence_transformer, embeddings, sentences, prompt_sentence, 10)

        col2.text_area(label='Semantic search output', value=best_sentences, height = 600)

    # for i in range(inputpdf.numPages):
    #     output = PdfFileWriter()
    #     output.addPage(inputpdf.getPage(i))
    #
    #     buf = io.BytesIO()
    #     output.write(buf)
    #     buf.seek(0)
    #
    #     base64_pdf = base64.b64encode(buf.read()).decode('utf-8')
    #     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#zoom=130" width="100%" height="600" type="application/pdf"></iframe>'
    #     col1.markdown(pdf_display, unsafe_allow_html=True)
    #
    #     buf.seek(0)
    #
    #     text = extract_text_from_pdf(buf.read())
    #     text = '\n'.join(text)
    #
    #     col2.text_area(label='', value=text, height=550)

    # inputpdf.getPage(0)
    #
    # base64_pdf = base64.b64encode(file_pdf.getvalue()).decode('utf-8')
    #
    # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    # st.markdown(pdf_display, unsafe_allow_html=True)