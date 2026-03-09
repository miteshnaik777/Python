import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample training sentences
data = [
    "how are you",
    "how are they",
    "how are we",
    "i am going to school",
    "i am going to office",
    "i am going to market",
    "thank you very much",
    "thank you so much",
    "see you tomorrow",
    "see you soon",
    "my name is Mitesh",
    "my name is Sharadha"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []

for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Build model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Function to predict next word
def predict_next_words(text, n=3):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len-1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)[0]

    top_indices = predicted.argsort()[-n:][::-1]

    predicted_words = [tokenizer.index_word[i] for i in top_indices]

    return predicted_words


# Take input from user
user_input = input("Enter a sentence: ")

prediction = predict_next_words(user_input)

print("Next word prediction:", prediction)

