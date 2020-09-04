from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()
model.add(
    Embedding(input_dim=num_words,
              input_length=training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True)
)

model.add(Masking(mask_value=0.0))

model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_words, activation='softmax'))

model.compile(optimize='adam', loss='categorical_crossentropy', metrics=['accuracy'])