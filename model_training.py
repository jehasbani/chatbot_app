import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
import random
from matplotlib import pyplot as plt

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('intents.json').read()
intents = json.loads(data_file)

# Tokenization
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append([w, intent['tag']])

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training a testing data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
print(training.shape)

X_train = list(training[:, 0])
y_train = list(training[:, 1])
print('Training data created')

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# adam = Adam()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

# plot training curves
fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

ax0.plot(hist.history['accuracy'])
ax0.set_title('Model accuracy')
ax0.set_ylabel('accuracy')
ax0.set_xlabel('epoch')
ax0.legend(['train', 'valid'], loc='upper left')

ax1.plot(hist.history['loss'])
ax1.set_title('Model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'valid'], loc='upper left')
plt.show()

print('Model created.')

