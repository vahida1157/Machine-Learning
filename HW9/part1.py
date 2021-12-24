import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

groups = fetch_20newsgroups()
np.unique(groups.target)

sns.histplot(groups.target)
plt.show()

count_vector = CountVectorizer(max_features=500)
data_count = count_vector.fit_transform(groups.data)
print(data_count[0])
data_count.toarray()
print(data_count.toarray()[0])
print(count_vector.get_feature_names())

data_cleaned = []
for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split()
                           if word.isalpha())
    data_cleaned.append(doc_cleaned)
print(data_cleaned)


def create_date_cleaned_count(paper_groups):
    all_names = set(names.words())
    count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

    lemmatizer = WordNetLemmatizer()
    data_cleaned = []
    for doc in paper_groups.data:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                               for word in doc.split()
                               if word.isalpha() and
                               word not in all_names)
        data_cleaned.append(doc_cleaned)

    data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)
    return data_cleaned_count


def create_tsne_model_and_show_plot(paper_groups):
    tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)

    data_tsne = tsne_model.fit_transform(create_date_cleaned_count(paper_groups).toarray())

    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=paper_groups.target)
    plt.show()


categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups_3 = fetch_20newsgroups(categories=categories_3)
create_tsne_model_and_show_plot(groups_3)

categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'comp.windows.x']
groups_5 = fetch_20newsgroups(categories=categories_5)
create_tsne_model_and_show_plot(groups_5)
