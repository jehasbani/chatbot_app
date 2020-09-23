import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, input_words, show_details=True):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(input_words)
    for sw in sentence_words:
        for i, w in enumerate(input_words):
            if w == sw:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, nn_model):
    p = bag_of_words(sentence, words, show_details=False)
    res = nn_model.predict(np.array([p]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    intents_list = intents_json['intents']
    for i in intents_list:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def bot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res


# Creating GUI with tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = bot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


app = Tk()
app.title = 'Hello'
app.geometry('400x500')
app.resizable(width=FALSE, height=FALSE)

ChatLog = Text(app, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(app, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(app,
                    font=("Verdana", 12, 'bold'),
                    text="Send",
                    width="12", height=5,
                    bd=0, bg="#32de97",
                    activebackground="#3c9d9b",
                    fg='#ffffff',
                    command=send)

EntryBox = Text(app, bd=0, bg="white", width="29", height="5", font="Arial")
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
app.mainloop()
