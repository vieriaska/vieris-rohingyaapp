import streamlit as st
import pandas as pd
import numpy as np
import json

import string
import re

import nltk
import preprocessor as p
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

from wordcloud import WordCloud
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

nltk.download('punkt')
tokenizer = Tokenizer()


# set_page_config adalah sebuah metode yang digunakan untuk mengubah parameter dari page kita
st.set_page_config(
    page_title="Sentimen Analysis Rohingya App",
    layout="wide"

)
st.title("Analisis Sentimen Masyarakat Indonesia terhadap Orang Rohingya berdasarkan Media Sosial Twitter")

path_dataset = st.secrets.path_configuration.path_dataset
filename = "rohingya.csv"

df = pd.read_csv(f"{path_dataset}{filename}")


stemfactory = StemmerFactory()
stemmer = stemfactory.create_stemmer()

pattern = r'[0-9]'
def preprocessing(text):
    for punctuation in string.punctuation:
        text = p.clean(text) #menghapus tag, hashtag
        text = re.sub(r'http[s]?://\S+','',text) #menghapus URL
        text = text.replace(punctuation, '') #menghapus tanda baca
        text = re.sub(pattern, '', text)#menghapus angka
        text = re.sub(r'\r?\n|\r','',text)#menghapus baris baru
        text = text.encode('ascii', 'ignore').decode('ascii') #menghapus emoji
        text = text.lower() #mengubah ke huruf kecil (case folding)
        text = stemmer.stem(text)
    return text

df['full_text'] = df['full_text'].astype(str)
df['processed'] = df['full_text'].apply(preprocessing)

# Normalisasi
with open('combined_slang_words.txt') as f:
    data = f.read()
dict_slang = json.loads(data)

# Fungsi untuk mengganti kata-kata dalam kalimat dengan value dari dictionary
def ganti_kata(kalimat, dictionary):
    kata_terpisah = kalimat.split()
    kata_baru = [dictionary.get(kata, kata) for kata in kata_terpisah]
    return " ".join(kata_baru)

# Menyamakan kata dalam kolom 'kalimat' dengan key pada dictionary dan mengganti dengan value pada dictionary
df['processed'] = df['processed'].apply(lambda x: ganti_kata(x, dict_slang))


factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def stopWord(text):
  for punctuation in string.punctuation:
    text = stopword.remove(text)
  return text

df['processed'] = df['processed'].astype(str)
df['processed'] = df['processed'].apply(stopWord)
df.head(10)

# Load lexicon
positive_lexicon = pd.read_csv("positive.tsv", sep="\t")
negative_lexicon = pd.read_csv("negative.tsv", sep="\t")
lexicon = pd.concat([positive_lexicon, negative_lexicon])

df['weight'] = df['processed'].apply(lambda x: sum(lexicon[lexicon['word'].isin(str(x).split())]['weight']))
df['label'] = df['weight'].apply(lambda x: 'netral' if x == 0 else ('negatif' if x < 0 else 'positif'))

st.header("Tabel Clean Berisikan Kalimat tentang Masyarakat Rohingya oleh Warga Twitter", divider="grey") 
st.caption('''
Nama: Vieri Aska Juneio Sembiring
''')
df = pd.DataFrame(df)
st.dataframe(df,hide_index=True)

# # Visualisasi dengan diagram batang
st.header("Diagram Batang Data", divider="grey")
# Hitung frekuensi setiap kategori label
label_counts = df['label'].value_counts()
# Hitung frekuensi setiap kategori label
label_counts = df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']

# Tentukan warna untuk setiap kategori label dengan kode HEX
colors = {'Negatif': '#FF5733', 'Positif': '#FFFF00', 'Netral': '#00FF00'}
label_counts['color'] = label_counts['label'].map(colors)

# Buat diagram batang interaktif dengan warna yang disesuaikan
fig = px.bar(label_counts, x='label', y='count', color='label', color_discrete_map=colors)

# Bagi layar menjadi dua kolom
col1, col2 = st.columns([3, 2])

# Tampilkan diagram di kolom pertama
with col1:
    st.plotly_chart(fig)

# Tampilkan keterangan di kolom kedua
with col2:
    keterangan = "Gambar disamping kiri merupakan sentimen analisis oleh masyarakat Twitter terhadap warga rohingya secara general. Sentimen Analisis tersebut menampilkan ketidaksukaan masyarakat Twitter terhadap rohingya dikarenakan sentimen Negatif memiliki jumlah tertinggi yaitu berjumlah 253."
    # st.caption(keterangan)
    st.caption(
        f'<p style="font-family: Roboto; color: black; font-size: 30px;">{keterangan}</p>', unsafe_allow_html=True
    )



# Definisikan fungsi untuk menampilkan WordCloud
def show_wordcloud(data, label):
    # Konversi nilai ke string menggunakan str()
    words = ' '.join(str(text) for text in data[data['label'] == label]['processed'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

    # Tampilkan wordcloud pada sumbu yang diberikan
    st.image(wordcloud.to_array(), caption=f'WordCloud for {label} Reviews', use_column_width=True)

# Panggil fungsi show_wordcloud untuk setiap label dan tampilkan dalam tiga kolom
unique_labels = df['label'].unique()

col1, col2, col3 = st.columns(3)

with col1:
    show_wordcloud(df, unique_labels[0])
    caption_1 = "Menampilkan kata kunci yang sering muncul dalam sentimen Negatif, yaitu Rohingya"
    st.caption(caption_1)
with col2:
    show_wordcloud(df, unique_labels[1])
    caption_2 = "Menampilkan kata kunci yang sering muncul dalam sentimen Netral, yaitu Rohingya"
    st.caption(caption_2)
with col3:
    show_wordcloud(df, unique_labels[2])
    caption_3 = "Menampilkan kata kunci yang sering muncul dalam sentimen Positif, yaitu Rohingya"
    st.caption(caption_3)
# SPLIT DATASET
# Bagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(df['processed'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

print('Train Dataset:')
print(len(y_train))
print(y_train.value_counts())
print('\n')

print('Validation Dataset:')
print(len(y_test))
print(y_test.value_counts())
print('\n')

# # DIAGRAM BATANG LATIH
# Hitung frekuensi setiap kategori label
label_counts_train = y_train.value_counts().reset_index()
label_counts_train.columns = ['label', 'count']

# Tentukan warna untuk setiap kategori label dengan kode HEX
colors_train = {'Negatif': '#FF5733', 'Positif': '#FFFF00', 'Netral': '#00FF00'}
label_counts_train['color'] = label_counts_train['label'].map(colors_train)

# # DIAGRAM BATANG UJI
# Hitung frekuensi setiap kategori label
label_counts_test = y_test.value_counts().reset_index()
label_counts_test.columns = ['label', 'count']

# Tentukan warna untuk setiap kategori label dengan kode HEX
colors_test = {'Negatif': '#FF5733', 'Positif': '#FFFF00', 'Netral': '#00FF00'}
label_counts_test['color'] = label_counts_test['label'].map(colors_test)

# Bagi layar menjadi dua kolom
col4, col5 = st.columns(2)

# Tampilkan diagram batang data latih di kolom pertama
with col4:
    st.header("Diagram Batang Data Latih", divider="grey")
    # Buat diagram batang interaktif dengan warna yang disesuaikan untuk data latih
    fig_train = px.bar(label_counts_train, x='label', y='count', color='label', color_discrete_map=colors_train)
    st.plotly_chart(fig_train)
    st.caption("Terdapat lebih banyak sampel untuk kategori Negatif (202) dibandingkan dengan Netral (77) dan Positif (70).")

# Tampilkan diagram batang data uji di kolom kedua
with col5:
    st.header("Diagram Batang Data Uji", divider="grey")
    # Buat diagram batang interaktif dengan warna yang disesuaikan untuk data uji
    fig_test = px.bar(label_counts_test, x='label', y='count', color='label', color_discrete_map=colors_test)
    st.plotly_chart(fig_test)
    st.caption("Distribusi frekuensi label dalam data uji mencerminkan distribusi yang serupa dengan data latih, dengan jumlah sampel terbanyak untuk kategori Negatif (51) dan jumlah sampel terendah untuk kategori Positif (17).")

caption_4 = "Data latih dan data uji menunjukkan ketidakseimbangan distribusi label, dengan kategori Negatif mendominasi."
st.markdown(f'<p style="font-family: Roboto; color: black; font-size: 30px;">{caption_4}</p>', unsafe_allow_html=True)


# Ekstraksi fitur menggunakan CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# TRAINING DATASET
# Model Naive Bayes
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_vectorized, y_train)
y_pred_nb = naive_bayes_classifier.predict(X_test_vectorized)

# Confusion Matrix, Recall, Precision, F1 Score untuk Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print("Confusion Matrix Naive Bayes:")
print(conf_matrix_nb)
print("\nClassification Report Naive Bayes:")
print(report_nb)


# # Membuat visualisasi confusion matrix
st.header("Confusion Matrix", divider="grey")
col6, col7 = st.columns(2)

with col6:
    
    fig = ff.create_annotated_heatmap(conf_matrix_nb, x=['Negatif', 'Positif', 'Netral'],
                                    y=['Negatif', 'Positif', 'Netral'], colorscale='Blues')

    # Menambahkan judul dan label
    fig.update_layout(title_text='Confusion Matrix',
                    xaxis=dict(title='Prediksi'),
                    yaxis=dict(title='Aktual'))

    # Menampilkan diagram 
    st.plotly_chart(fig)

with col7:
    caption_5 = '''
    Berdasarkan confusion matrix, berikut beberapa kesimpulan:
    Model lebih akurat dalam memprediksi kategori "Negatif" dibandingkan "Netral" dan "Positif".
    Model cenderung salah memprediksi sampel "Netral" sebagai "Positif".
    Model cenderung salah memprediksi sampel "Positif" sebagai "Netral" dan "Negatif".
    '''
    st.caption(
        f'<p style="font-family: Roboto; color: black; font-size: 30px;">{caption_5}</p>', unsafe_allow_html=True
    )
    

# Menambahkan kolom prediksi dan label aktual ke DataFrame uji
st.header("Tabel Kolom Prediksi dan Aktual berdasarkan Sentimennya", divider="grey") 
col8, col9 = st.columns(2)  
with col8: 
    df_uji = pd.DataFrame({'Text': X_test, 'Label Aktual': y_test, 'Label Prediksi': y_pred_nb})
    st.dataframe(df_uji.head(50),hide_index=True)
with col9:
    caption_6 = '''
    Tabel di samping kiri menampilkan Machine Learning Sentimen Analysis dalam mendapatkan nilai aktualnya berdasarkan data yang diprediksi.
    Sentimen Negatif mendominasi, dikarenakan banyaknya cibiran, kalimat, dan kata yang bersifat negatif terhadap masyarakat Rohingya oleh 
    Warga Twitter.  
    '''
    st.caption(
        f'<p style="font-family: Roboto; color: black; font-size: 30px;">{caption_6}</p>', unsafe_allow_html=True
    )

