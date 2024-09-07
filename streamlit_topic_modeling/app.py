import random

import gensim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from wordcloud import WordCloud

DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001
DEFAULT_NUM_TOPICS = 6

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


def lda_options():
    return {
        'num_topics': st.number_input('Number of Topics', min_value=1, value=9,
                                      help='The number of requested latent topics to be extracted from the training corpus.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Number of documents to be used in each training chunk.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Number of passes through the corpus during training.'),
        'update_every': st.number_input('Update Every', min_value=1, value=1,
                                        help='Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.'),
        'alpha': st.selectbox('𝛼', ('symmetric', 'asymmetric', 'auto'),
                              help='A priori belief on document-topic distribution.'),
        'eta': st.selectbox('𝜂', (None, 'symmetric', 'auto'), help='A-priori belief on topic-word distribution'),
        'decay': st.number_input('𝜅', min_value=0.5, max_value=1.0, value=0.5,
                                 help='A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined.'),
        'offset': st.number_input('𝜏_0', value=1.0,
                                  help='Hyper-parameter that controls how much we will slow down the first steps the first few iterations.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='Log perplexity is estimated every that many updates.'),
        'iterations': st.number_input('Iterations', min_value=1, value=50,
                                      help='Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.'),
        'gamma_threshold': st.number_input('𝛾', min_value=0.0, value=0.001,
                                           help='Minimum change in the value of the gamma parameters to continue iterating.'),
        'minimum_probability': st.number_input('Minimum Probability', min_value=0.0, max_value=1.0, value=0.01,
                                               help='Topics with a probability lower than this threshold will be filtered out.'),
        'minimum_phi_value': st.number_input('𝜑', min_value=0.0, value=0.01,
                                             help='if per_word_topics is True, this represents a lower bound on the term probabilities.'),
        'per_word_topics': st.checkbox('Per Word Topics',
                                       help='If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).')
    }


def nmf_options():
    return {
        'num_topics': st.number_input('Number of Topics', min_value=1, value=9, help='Number of topics to extract.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Number of documents to be used in each training chunk.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Number of full passes over the training corpus.'),
        'kappa': st.number_input('𝜅', min_value=0.0, value=1.0, help='Gradient descent step size.'),
        'minimum_probability': st.number_input('Minimum Probability', min_value=0.0, max_value=1.0, value=0.01,
                                               help='If normalize is True, topics with smaller probabilities are filtered out. If normalize is False, topics with smaller factors are filtered out. If set to None, a value of 1e-8 is used to prevent 0s.'),
        'w_max_iter': st.number_input('W max iter', min_value=1, value=200,
                                      help='Maximum number of iterations to train W per each batch.'),
        'w_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.0001,
                                            help=' If error difference gets less than that, training of W stops for the current batch.'),
        'h_max_iter': st.number_input('H max iter', min_value=1, value=50,
                                      help='Maximum number of iterations to train h per each batch.'),
        'h_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.001,
                                            help='If error difference gets less than that, training of h stops for the current batch.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='Number of batches after which l2 norm of (v - Wh) is computed.'),
        'normalize': st.selectbox('Normalize', (True, False, None), help='Whether to normalize the result.')
    }


MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': gensim.models.LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    },
    'Non-Negative Matrix Factorization': {
        'options': nmf_options,
        'class': gensim.models.Nmf,
        'help': 'https://radimrehurek.com/gensim/models/nmf.html'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

WORDCLOUD_FONT_PATH = r'./data/Inkfree.ttf'

EMAIL_REGEX_STR = r'\S*@\S*'
MENTION_REGEX_STR = r'@\S*'
HASHTAG_REGEX_STR = r'#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


@st.cache_data()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')


@st.cache_data()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', text) for text in texts]
    docs = [[w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('english')] for doc in texts]
    return docs


@st.cache_data()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs


@st.cache_data()
def create_trigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[docs])
    trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs


@st.cache_data()
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)
    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    if ngrams == 'trigrams':
        docs = create_trigrams(docs)
    return docs


@st.cache_data()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                          background_color='white', collocations=collocations).generate(wordcloud_text)
    return wordcloud


@st.cache_data()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


@st.cache_data()
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model


def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model', 'previous_perplexity', 'previous_coherence_model_value'):
        if key in st.session_state:
            del st.session_state[key]


def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))


def calculate_coherence(model, corpus, coherence):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()


@st.cache_data()
def white_or_black_text(background_color):
    # https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'


def perplexity_section():
    with st.spinner('Calculating Perplexity ...'):
        perplexity = calculate_perplexity(st.session_state.model, st.session_state.corpus)
    key = 'previous_perplexity'
    delta = f'{perplexity - st.session_state[key]:.4}' if key in st.session_state else None
    st.metric(label='Perplexity', value=f'{perplexity:.4f}', delta=delta, delta_color='inverse')
    st.session_state[key] = perplexity
    st.markdown('Viz., https://en.wikipedia.org/wiki/Perplexity')
    st.latex(r'Perplexity = \exp\left(-\frac{\sum_d \log(p(w_d|\Phi, \alpha))}{N}\right)')


def coherence_section():
    with st.spinner('Calculating Coherence Score ...'):
        coherence = calculate_coherence(st.session_state.model, st.session_state.corpus, 'u_mass')
    key = 'previous_coherence_model_value'
    delta = f'{coherence - st.session_state[key]:.4f}' if key in st.session_state else None
    st.metric(label='Coherence Score', value=f'{coherence:.4f}', delta=delta)
    st.session_state[key] = coherence
    st.markdown('Viz., http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf')
    st.latex(
        r'C_{UMass} = \frac{2}{N \cdot (N - 1)}\sum_{i=2}^N\sum_{j=1}^{i-1}\log\frac{P(w_i, w_j) + \epsilon}{P(w_j)}')


@st.cache_data()
def train_projection(projection, n_components, df):
    if projection == 'PCA':
        projection_model = PCA(n_components=n_components)
    elif projection == 'T-SNE':
        projection_model = TSNE(n_components=n_components)
    elif projection == 'UMAP':
        projection_model = UMAP(n_components=n_components)
    else:
        raise ValueError(f'Unknown projection: {projection}')
    return projection_model.fit_transform(df)


if __name__ == '__main__':
    st.set_page_config(page_title='Topic Modeling', page_icon='./data/favicon.png', layout='wide')

    preprocessing_options = st.sidebar.form('preprocessing-options')
    with preprocessing_options:
        st.header('Preprocessing Options')
        ngrams = st.selectbox('N-grams', [None, 'bigrams', 'trigams'], help='TODO ...')  # TODO ...
        st.form_submit_button('Preprocess')

    visualization_options = st.sidebar.form('visualization-options')
    with visualization_options:
        st.header('Visualization Options')
        collocations = st.checkbox('Enable WordCloud Collocations',
                                   help='Collocations in word clouds enable the display of phrases.')
        highlight_probability_minimum = st.select_slider('Highlight Probability Minimum',
                                                         options=[10 ** exponent for exponent in range(-10, 1)],
                                                         value=DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
                                                         help='Minimum topic probability in order to color highlight a word in the _Topic Highlighted Sentences_ visualization.')
        st.form_submit_button('Apply')

    st.title('Topic Modeling')
    st.header('What is topic modeling?')
    with st.expander('Hero Image'):
        st.image('./data/is-this-a-topic-modeling.jpg', caption='No ... no it\'s not ...', use_column_width=True)
    st.markdown(
        'Topic modeling is a broad term. It encompasses a number of specific statistical learning methods. '
        'These methods do the following: explain documents in terms of a set of topics and those topics in terms of '
        'the a set of words. Two very commonly used methods are Latent Dirichlet Allocation (LDA) and Non-Negative '
        'Matrix Factorization (NMF), for instance. Used without additional qualifiers the approach is usually assumed '
        'to be unsupervised although there are semi-supervised and supervised variants.'
    )

    with st.expander('Additional Details'):
        st.markdown('The objective can be viewed as a matrix factorization.')
        st.image('./data/mf.png', use_column_width=True)
        st.markdown('This factorization makes the methods much more efficient than directly characterizing documents '
                    'in term of words.')
        st.markdown('More information on LDA and NMF can be found at '
                    'https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation and '
                    'https://en.wikipedia.org/wiki/Non-negative_matrix_factorization, respectively.')

    st.header('Datasets')
    st.markdown('Preloaded a couple of small example datasets to illustrate.')
    selected_dataset = st.selectbox('Dataset', [None, *sorted(list(DATASETS.keys()))], on_change=clear_session_state)
    if not selected_dataset:
        st.write('Choose a Dataset to Conintue ...')
        st.stop()

    with st.expander('Dataset Description'):
        st.markdown(DATASETS[selected_dataset]['description'])
        st.markdown(DATASETS[selected_dataset]['url'])

    text_column = DATASETS[selected_dataset]['column']
    texts_df = generate_texts_df(selected_dataset)
    docs = generate_docs(texts_df, text_column, ngrams=ngrams)

    with st.expander('Sample Documents'):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    with st.expander('Frequency Sized Corpus Wordcloud'):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(), caption='Dataset Wordcloud (Not A Topic Model)', use_column_width=True)
        st.markdown('These are the remaining words after document preprocessing.')

    with st.expander('Document Word Count Distribution'):
        len_docs = [len(doc) for doc in docs]
        fig, ax = plt.subplots()
        sns.histplot(data=pd.DataFrame(len_docs, columns=['Words In Document']), discrete=True, ax=ax)
        st.pyplot(fig)

    model_key = st.sidebar.selectbox('Model', [None, *list(MODELS.keys())], on_change=clear_session_state)
    model_options = st.sidebar.form('model-options')
    if not model_key:
        with st.sidebar:
            st.write('Choose a Model to Continue ...')
        st.stop()
    with model_options:
        st.header('Model Options')
        model_kwargs = MODELS[model_key]['options']()
        st.session_state['model_kwargs'] = model_kwargs
        train_model_clicked = st.form_submit_button('Train Model')

    if train_model_clicked:
        with st.spinner('Training Model ...'):
            id2word, corpus, model = train_model(docs, MODELS[model_key]['class'], **st.session_state.model_kwargs)
        st.session_state.id2word = id2word
        st.session_state.corpus = corpus
        st.session_state.model = model

    if 'model' not in st.session_state:
        st.stop()

    st.header('Model')
    st.write(type(st.session_state.model).__name__)
    st.write(st.session_state.model_kwargs)

    st.header('Model Results')

    topics = st.session_state.model.show_topics(formatted=False, num_words=50,
                                                num_topics=st.session_state.model_kwargs['num_topics'], log=False)
    with st.expander('Topic Word-Weighted Summaries'):
        topic_summaries = {}
        for topic in topics:
            topic_index = topic[0]
            topic_word_weights = topic[1]
            topic_summaries[topic_index] = ' + '.join(
                f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
        for topic_index, topic_summary in topic_summaries.items():
            st.markdown(f'**Topic {topic_index}**: _{topic_summary}_')

    colors = random.sample(COLORS, k=model_kwargs['num_topics'])
    with st.expander('Top N Topic Keywords Wordclouds'):
        cols = st.columns(3)
        for index, topic in enumerate(topics):
            wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                           background_color='white', collocations=collocations, prefer_horizontal=1.0,
                           color_func=lambda *args, **kwargs: colors[index])
            with cols[index % 3]:
                wc.generate_from_frequencies(dict(topic[1]))
                st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

    with st.expander('Topic Highlighted Sentences'):
        sample = texts_df.sample(10)
        for index, row in sample.iterrows():
            html_elements = []
            for token in row[text_column].split():
                if st.session_state.id2word.token2id.get(token) is None:
                    html_elements.append(f'<span style="text-decoration:line-through;">{token}</span>')
                else:
                    term_topics = st.session_state.model.get_term_topics(token, minimum_probability=0)
                    topic_probabilities = [term_topic[1] for term_topic in term_topics]
                    max_topic_probability = max(topic_probabilities) if topic_probabilities else 0
                    if max_topic_probability < highlight_probability_minimum:
                        html_elements.append(token)
                    else:
                        max_topic_index = topic_probabilities.index(max_topic_probability)
                        max_topic = term_topics[max_topic_index]
                        background_color = colors[max_topic[0]]
                        # color = 'white'
                        color = white_or_black_text(background_color)
                        html_elements.append(
                            f'<span style="background-color: {background_color}; color: {color}; opacity: 0.5;">{token}</span>')
            st.markdown(f'Document #{index}: {" ".join(html_elements)}', unsafe_allow_html=True)

    has_log_perplexity = hasattr(st.session_state.model, 'log_perplexity')
    with st.expander('Metrics'):
        if has_log_perplexity:
            left_column, right_column = st.columns(2)
            with left_column:
                perplexity_section()
            with right_column:
                coherence_section()
        else:
            coherence_section()

    with st.expander('Low Dimensional Projections'):
        with st.form('projections-form'):
            left_column, right_column = st.columns(2)
            projection = left_column.selectbox('Projection', ['PCA', 'T-SNE', 'UMAP'], help='TODO ...')
            plot_type = right_column.selectbox('Plot', ['2D', '3D'], help='TODO ...')
            n_components = 3
            columns = [f'proj{i}' for i in range(1, 4)]
            generate_projection_clicked = st.form_submit_button('Generate Projection')

        if generate_projection_clicked:
            topic_weights = []
            for index, topic_weight in enumerate(st.session_state.model[st.session_state.corpus]):
                weight_vector = [0] * int(st.session_state.model_kwargs['num_topics'])
                for topic, weight in topic_weight:
                    weight_vector[topic] = weight
                topic_weights.append(weight_vector)
            df = pd.DataFrame(topic_weights)
            dominant_topic = df.idxmax(axis='columns').astype('string')
            dominant_topic_percentage = df.max(axis='columns')
            df = df.assign(dominant_topic=dominant_topic, dominant_topic_percentage=dominant_topic_percentage,
                           text=texts_df[text_column])
            with st.spinner('Training Projection'):
                projections = train_projection(projection, n_components, df.drop(columns=['dominant_topic', 'dominant_topic_percentage', 'text']).add_prefix('topic_'))
            data = pd.concat([df, pd.DataFrame(projections, columns=columns)], axis=1)

            px_options = {'color': 'dominant_topic', 'size': 'dominant_topic_percentage',
                          'hover_data': ['dominant_topic', 'dominant_topic_percentage', 'text']}
            if plot_type == '2D':
                fig = px.scatter(data, x='proj1', y='proj2', **px_options)
                st.plotly_chart(fig)
                fig = px.scatter(data, x='proj1', y='proj3', **px_options)
                st.plotly_chart(fig)
                fig = px.scatter(data, x='proj2', y='proj3', **px_options)
                st.plotly_chart(fig)
            elif plot_type == '3D':
                fig = px.scatter_3d(data, x='proj1', y='proj2', z='proj3', **px_options)
                st.plotly_chart(fig)

    if hasattr(st.session_state.model, 'inference'):  # gensim Nmf has no 'inference' attribute so pyLDAvis fails
        if st.button('Generate pyLDAvis'):
            with st.spinner('Creating pyLDAvis Visualization ...'):
                py_lda_vis_data = pyLDAvis.gensim_models.prepare(st.session_state.model, st.session_state.corpus,
                                                                 st.session_state.id2word)
                py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
            with st.expander('pyLDAvis', expanded=True):
                st.markdown('pyLDAvis is designed to help users interpret the topics in a topic model that has been '
                            'fit to a corpus of text data. The package extracts information from a fitted LDA topic '
                            'model to inform an interactive web-based visualization.')
                st.markdown('https://github.com/bmabey/pyLDAvis')
                components.html(py_lda_vis_html, width=1300, height=800)
