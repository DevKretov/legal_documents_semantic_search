import streamlit as st
import pandas as pd
import base64
#from PyPDF2 import PdfFileWriter, PdfFileReader
#from pdf2image import convert_from_path
import io
import unicodedata
import re
from typing import List

from lexnlp.extract.en.entities.nltk_maxent import get_company_annotations
import lexnlp.extract.en.durations
import lexnlp.extract.en.definitions
import lexnlp.extract.en.acts
import lexnlp.extract.en.dates
import lexnlp.extract.en.money

import logging
logger = logging.getLogger('stanza').setLevel(logging.WARNING)

import fitz

from sentence_transformers import SentenceTransformer
from sentence_transformers import util

CLAUSES_REGEX = re.compile(r"((\d\.?)+R?)+")

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

def correct_clauses_all_sentences(all_sentences):
    all_sentences_new = []
    for i in range(len(all_sentences)):
        sentence = all_sentences[i]
        matches = CLAUSES_REGEX.finditer(sentence)
        for match in matches:

            if match and len(match.group()) == len(sentence.strip()) and i + 1 != len(all_sentences):
                # then it's clause
                all_sentences[i + 1] = match.group() + ' ' + all_sentences[i + 1]
                all_sentences[i] = None
                break

            match_len = len(match.group())
            match_index = sentence.index(match.group())

            if match and match_index + match_len == len(sentence) and match_index != 0 and i + 1 != len(all_sentences):
                # then it's the end of the sentence and it has to go to the next sentence
                all_sentences[i + 1] = match.group() + ' ' + all_sentences[i + 1]
                all_sentences[i] = all_sentences[i][:match_index].strip()
                break

    all_sentences = list(filter(lambda _block: _block is not None, all_sentences))
    return all_sentences

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

    text_blocks_all = [' '.join(block.split()).strip() for block in text_blocks_all]

    # def reconstruct_sentences(text_block):

    text_block = text_blocks_all

    for i in range(len(text_block) - 1, 0, -1):
        previous_text = text_block[i - 1].strip()
        current_text = text_block[i]

        last_char = previous_text[-1] if len(previous_text) != 0 else None
        if last_char is None:
            continue

        if last_char.isalnum() or last_char in [',', ':', '-', '(', ')', '"', '\'']:
            # if it's a part of a text
            previous_text = text_block[i - 1].strip() + ' ' + current_text
            text_block[i - 1] = previous_text

            text_block[i] = None

    text_block_filtered = list(filter(lambda _block: _block is not None, text_block))
    all_sentences = []

    for _block in text_block_filtered:
        all_sentences.extend(stanza_sent_tokenize(_block))

    all_sentences = list(map(lambda _sentence: ' '.join(_sentence.split()).strip(), all_sentences))
    all_sentences = correct_clauses_all_sentences(all_sentences)

    for i, _sentence in enumerate(all_sentences[:30]):
        print(f'Sentence {i}: {_sentence}')

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
    pdf_display = f'<p align="center"><iframe src="data:application/pdf;base64,{base64_pdf}#zoom=130" width="70%" height="600" type="application/pdf"></iframe></p>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    tabs = st.tabs(["Semantic search", 'Definitions', 'Companies', "Contraints", 'Acts', 'Dates', 'Money', 'Text'])

    tab_semantic = tabs[0]
    tab_definitions = tabs[1]
    tab_companies = tabs[2]
    tab_extractions = tabs[3]
    tab_acts = tabs[4]
    tab_dates = tabs[5]
    tab_money = tabs[6]
    tab_text = tabs[-1]

    # with tab_semantic:
    col1, col2 = tab_semantic.columns(2)
    form = col1.form(key='my-form')
    prompt_sentence = form.text_input('Enter prompt sentence')
    submit = form.form_submit_button('Find')


    if submit:
        best_sentences = find_top_n_sentences_to_prompt(sentence_transformer, embeddings, sentences, prompt_sentence, 10)

        col2.text_area(label='Semantic search output', value=best_sentences, height = 600)

    with tab_text:
        st.text('\n'.join(sentences))

    with tab_dates:
        for i in range(len(sentences)):
            entry = sentences[i]
            extractions = list(lexnlp.extract.en.dates.get_dates(entry))
            if len(extractions) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):
                for extraction in extractions:
                    datetime_obj = extraction
                    datetime_str = datetime_obj.strftime("%d %b, %Y")

                    col_datetime, col_text = st.columns([2, 5])
                    col_datetime.text(datetime_str)

                    col_text.markdown(entry)

        # .strftime("%d %b, %Y")

    with tab_money:
        constraints_list = []
        for i in range(len(sentences)):
            entry = sentences[i]
            moneys = list(lexnlp.extract.en.money.get_money(entry))
            if len(moneys) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):
                for money in moneys:
                    money_amount, money_currency = money
                    money_amount = str(money_amount)
                    col_money, col_text = st.columns([2, 5])
                    col_money.text(money_amount + ' ' + money_currency)

                    col_text.markdown(entry)
                    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    with tab_definitions:
        constraints_list = []
        for i in range(len(sentences)):
            entry = sentences[i]
            definitions = list(lexnlp.extract.en.definitions.get_definitions(entry))
            if len(definitions) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):
                for definition in definitions:
                    col_definitions, col_text = st.columns([2, 5])
                    col_definitions.text(definition)
                    markdown_text = re.sub(definition, f'<span style="color:red">{definition}</span>', entry)
                    col_text.markdown(markdown_text, unsafe_allow_html=True)
                    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    with tab_acts:
        extractions_list = []
        for i in range(len(sentences)):
            entry = sentences[i]
            extractions = list(lexnlp.extract.en.acts.get_acts(entry))
            if len(extractions) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):
                for extraction in extractions:
                    act_name = extraction['act_name']
                    loc_start, loc_end = extraction['location_start'], extraction['location_end']
                    col_extractions, col_text = st.columns([2, 5])
                    col_extractions.text(act_name)
                   # markdown_text = f'{entry[:loc_start]}<span style="color:red">{entry[loc_start:loc_end]}</span>{entry[loc_end:]}'
                    markdown_text = re.sub(act_name, f'<span style="color:red">{act_name}</span>', entry)
                    col_text.markdown(markdown_text, unsafe_allow_html=True)
                    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)




    with tab_companies:
        companies_list = []
        for i in range(len(sentences)):
            entry = sentences[i]
            companies = list(get_company_annotations(entry))
            if len(companies) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):

                for company in companies:
                    print(f'{company.name} - {entry}' )
                    company_name, span = company.name, company.coords
                    col_companies, col_text = st.columns([2, 5])
                    col_companies.text(company_name)
                    if company_name in entry:
                        start_index = entry.index(company_name)
                        end_index = start_index + len(company_name)

                        markdown_text = re.sub(company_name, f'<span style="color:red">{company_name}</span>', entry)
                       # markdown_text = f'{entry[:start_index]}<span style="color:red">{entry[start_index:end_index]}</span>{entry[end_index:]}'
                    else:
                        markdown_text = f'{entry[:span[0]]}<span style="color:red">{entry[span[0]:span[1]]}</span>{entry[span[1]:]}'
                    col_text.markdown(markdown_text, unsafe_allow_html=True)
                    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                                unsafe_allow_html=True)


    with tab_extractions:
        constraints_list = []
        for i in range(len(sentences)):
            entry = sentences[i]
            constraints = list(lexnlp.extract.en.durations.get_durations(entry))
            if len(constraints) > 0:  # and not ((i == 0) or ((i+1) == len(filtered_text))):
                for constraint in constraints:
                    col_constraints, col_days, col_text = st.columns([2, 2, 5])
                    col_constraints.text(str(constraint[1]) + ' ' + constraint[0])
                    col_days.text(constraint[-1])
                    col_text.markdown(entry)

              #      constraints_list.append((str(constraint[1]) + ' ' + constraint[0], constraint[-1], sentences[i]))

      #  df = pd.DataFrame(constraints_list, columns = ['Constraint', 'Days', 'Text extract', ])
      #  st.dataframe(df)
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