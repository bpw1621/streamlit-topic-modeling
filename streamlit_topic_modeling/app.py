import random

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import regex
import seaborn as sns
import spacy
import streamlit as st
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.colors as mcolors

nltk.download("stopwords")

DATASETS = {
    'Five Years of Elon Musk Tweets': {
        'path': './data/elonmusk.csv.zip',
        'column': 'tweet',
        'url': 'https://www.kaggle.com/vidyapb/elon-musk-tweets-2015-to-2020',
        'description': (
            'I scraped Elon Musk\'s tweets from the last 5 years using twint library. My inspiration behind this is to '
            'see how public personalities are influencing common people on Social Media Platforms. I would love to see '
            'some notebooks around this dataset, giving us insights like what are the topics which Tesla mostly tweets '
            'about? How are Tesla\'s stocks being influenced by his tweets?'
        )
    },
    'Airline Tweets': {
        'path': './data/Tweets.csv.zip',
        'column': 'text',
        'url': 'https://www.kaggle.com/crowdflower/twitter-airline-sentiment',
        'description': (
            'A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from '
            'February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, '
            'followed by categorizing negative reasons (such as "late flight" or "rude service").'
        )
    }
}

MODELS = {
    'lda': {
        'display_name': 'Latent Dirichlet Allocation (LDA)',
    },
    'nmf': {
        'display_name': 'Non-Negative Matrix Factorization (NMF)'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

WORDCLOUD_FONT_PATH = r'./data/Inkfree.ttf'

EMAIL_REGEX_STR = '\S*@\S*'
MENTION_REGEX_STR = '@\S*'
HASHTAG_REGEX_STR = '#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


@st.cache()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')


@st.cache()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', text) for text in texts]
    docs = [[w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('english')] for doc in texts]
    return docs


@st.cache()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs


@st.cache()
def create_trigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[docs])
    trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs


@st.cache()
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)

    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    if ngrams == 'trigrams':
        docs = create_trigrams(docs)

    lemmantized_docs = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for doc in docs:
        doc = nlp(' '.join(doc))
        # lemmantized_docs.append([token.lemma_ for token in doc if token.pos_ in ('NOUN', 'ADJ', 'VERB', 'ADV')])
        lemmantized_docs.append([token.lemma_ for token in doc])

    # lemmantized_docs = [[w for w in simple_preprocess(str(doc)) if w not in stopwords.words('english')] for doc in
    #                     lemmantized_docs]

    return lemmantized_docs


@st.cache()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                          background_color='white', collocations=collocations).generate(wordcloud_text)
    return wordcloud


# @st.cache()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


def train_model(docs, num_topics: int = 10, per_word_topics: bool = True):
    id2word, corpus = prepare_training_data(docs)
    model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                   per_word_topics=per_word_topics)
    return model


# LDA
# chunksize=2000, passes=1, update_every=1, alpha='symmetric', eta=None, decay=0.5,
# offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01, random_state=None,
# ns_conf=None, minimum_phi_value=0.01, per_word_topics=False

# NMF
# num_topics=100, chunksize=2000, passes=1, kappa=1.0, minimum_probability=0.01, w_max_iter=200,
# w_stop_condition=0.0001, h_max_iter=50, h_stop_condition=0.001, eval_every=10, normalize=True, random_state=None

if __name__ == '__main__':
    st.set_page_config(page_title='Topic Modeling', page_icon='./data/favicon.png')

    ngrams = None
    with st.sidebar:
        st.header('Settings')
        st.subheader('WordCloud')
        collocations = st.checkbox('Enable Collocations')
        st.markdown('<sup>Collocations in word clouds enable the display of phrases.</sup>', unsafe_allow_html=True)

        st.subheader('Preprocessing')
        if st.checkbox('Use N-grams'):
            ngrams = st.selectbox('N-grams', ['bigrams', 'trigams'])

        st.subheader('Topic Modeling')
        st.selectbox('Base Model', [model['display_name'] for model in MODELS.values()])
        num_topics = st.number_input('Number of Topics', min_value=1, max_value=200, value=10)
        per_word_topics = st.checkbox('Per Word Topics', value=True)

    st.title('Topic Modeling')
    st.header('What is topic modeling?')
    with st.beta_expander('Hero', expanded=True):
        st.image('./data/is-this-a-topic-modeling.jpg', caption='No ... no it\'s not ...', use_column_width=True)
    st.markdown(
        'Topic modeling is a broad term. It encompasses a number of specific statistical learning methods. '
        'These methods do the following: explain documents in terms of a set of topics and those topics in terms of '
        'the a set of words.'
    )
    st.markdown(
        'Two very commonly used methods are Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization '
        '(NMF), for instance.'
    )
    st.markdown(
        'Used without additional qualifiers the approach is usually assumed to be '
        'unsupervised although there are semi-supervised and supervised variants.'
    )

    with st.beta_expander('Additional Details'):
        st.markdown('The objective can be viewed as a matrix factorization.')
        st.image('./data/mf.png', use_column_width=True)
        st.markdown('This factorization makes the methods much more efficient than directly characterizing documents '
                    'in term of words.')
        st.markdown('LDA is ... TODO ...')  # TODO
        st.markdown('NMF is ... TODO ...')  # TODO

    st.header('Datasets')
    st.markdown('Preloaded a couple of small example datasets to illustrate.')
    selected_dataset = st.selectbox('Dataset', sorted(list(DATASETS.keys())))
    with st.beta_expander('Dataset Description', expanded=True):
        st.markdown(DATASETS[selected_dataset]['description'])
        st.markdown(DATASETS[selected_dataset]['url'])

    text_column = DATASETS[selected_dataset]['column']
    texts_df = generate_texts_df(selected_dataset)
    docs = generate_docs(texts_df, text_column, ngrams=ngrams)

    with st.beta_expander('Sample Documents', expanded=True):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    with st.beta_expander('Frequency Sized Corpus Wordcloud', expanded=True):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(), caption='Dataset Wordcloud (Not A Topic Model)', use_column_width=True)
        st.markdown('These are the remaining words after document preprocessing.')

    with st.beta_expander('Document Word Count Distribution', expanded=True):
        len_docs = [len(doc) for doc in docs]
        fig, ax = plt.subplots()
        sns.histplot(pd.DataFrame(len_docs, columns=['Words In Document']), ax=ax)
        st.pyplot(fig)

    model = None
    with st.sidebar:
        if st.button('Train'):
            model = train_model(docs, num_topics, per_word_topics)

    st.header('Model')
    if model:
        st.write(model)
    else:
        st.markdown('No model has been trained yet: use the sidebar to configure and train a topic modeling model.')

    st.header('Model Results')
    if model:
        topics = model.print_topics()
        st.subheader('Topic Word-Weighted Summaries')
        for topic in topics:
            st.markdown(f'**Topic #{topic[0]}**: _{topic[1]}_')

        st.subheader('Top N Topic Keywords Wordclouds')
        topics = model.show_topics(formatted=False, num_topics=num_topics)
        cols = st.beta_columns(3)
        colors = random.sample(COLORS, k=len(topics))
        for index, topic in enumerate(topics):
            wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                           background_color='white', collocations=collocations, prefer_horizontal=1.0,
                           color_func=lambda *args, **kwargs: colors[index])
            with cols[index % 3]:
                wc.generate_from_frequencies(dict(topic[1]))
                st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)
    else:
        st.markdown('No model has been trained yet: use the sidebar to configure and train a topic modeling model.')
