import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # 이 코드는 변경하지 마세요. 테스트가 작동하지 않을 수 있습니다.
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # 데이터셋 로드 및 전처리
    with open("sarcasm.json", 'r') as f:
        data = json.load(f)
        for item in data:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # 토크나이저 및 패딩 처리
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # 데이터 분할
    training_sentences = padded[:training_size]
    training_labels = np.array(labels[:training_size])
    testing_sentences = padded[training_size:]
    testing_labels = np.array(labels[training_size:])

    # 모델 구축
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 훈련
    history = model.fit(training_sentences, training_labels, epochs=100, validation_data=(testing_sentences, testing_labels), verbose=1)

    # 훈련 과정에서의 정확도 확인
    train_accuracy = history.history['accuracy']
    test_accuracy = history.history['val_accuracy']
    print("훈련 정확도:", train_accuracy)
    print("검증 정확도:", test_accuracy)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("sarcasm.h5")
