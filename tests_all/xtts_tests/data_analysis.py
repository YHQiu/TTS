import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 下载NLTK的punkt分词器模型
nltk.download('punkt')

def main():
    # 读取CSV文件
    file_path = 'train_data/ms/metadata.csv'  # 替换成你的文件路径
    data = pd.read_csv(file_path, sep='|', header=None, names=['ID', 'Text'])

    # 查看数据前几行
    print(data.head())

    # 分词
    data['Tokenized_Text'] = data['Text'].apply(lambda x: word_tokenize(x.lower()))  # 小写化并分词

    # 特征提取 - 使用词袋模型
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Text'])

    # 查看特征（词汇）列表
    feature_names = vectorizer.get_feature_names_out()
    print("特征数量:", len(feature_names))
    print("部分特征示例:", feature_names[:20])

    # 计算TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text'])

    # 计算余弦相似度作为特征之间的相关性
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 打印相关性矩阵
    print("相关性矩阵:\n", cosine_sim)

    # 绘制词云图表
    all_text = ' '.join(data['Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Text Data')
    plt.show()

if __name__ == "__main__":
    main()
