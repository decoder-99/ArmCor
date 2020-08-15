import os
import pickle
import scipy
from google_drive_downloader import GoogleDriveDownloader as gdd
from tensorflow.keras.models import model_from_json


file_description = {
    'word vectorizer': ['word_vectorizer.pickle', '1-5c-wlG4OTlkwwtwsPSclt-VzxaCVuhV'],
    'char vectorizer': ['char_vectorizer.pickle', '1-7IS7aJ-uPDAZDN3NhBkMTdVbi1gJuAi'],
    'detection model': ['detection_model.json', '13v3kXkJBdRl0M6APApcRnM0hxqofL8ZM'],
    'detection model weights': ['detection_model_weights.h5', '1-3caGT9ckPccnN1pl72FME6egk04sG7q'],
    'correction model': ['correction_model.pkl', '1P1meD8r0lRJAJcLxWMlCunq-qtFnsg4Q']
}

current_dir = __file__[:-7]
print(current_dir)

for file_info in file_description.values():
  if not os.path.isfile(current_dir+'/'+file_info[0]):
    save_path = current_dir + '/' + file_info[0]
    gdd.download_file_from_google_drive(file_info[1], save_path)

with open(current_dir + '/' + file_description['word vectorizer'][0], 'rb') as word_vec:
    word_vectorizer = pickle.load(word_vec)
with open(current_dir + '/' + file_description['char vectorizer'][0], 'rb') as char_vec:
    char_vectorizer = pickle.load(char_vec)

with open(current_dir + '/' + file_description['detection model'][0], 'rb') as json_file:
    detection_model = model_from_json(json_file.read())
detection_model.load_weights(current_dir + '/' + file_description['detection model weights'][0])


def detect_errors(tokens: list):
    predicted_labels = []
    for token in tokens:
        token_vec = scipy.sparse.hstack([word_vectorizer.transform([token]), char_vectorizer.transform([token])]).toarray()
        predicted_labels.append(detection_model.predict(token_vec)[0] > 0.5)
    return predicted_labels


def make_conllu(tokens: list, labels: list):
    with open(current_dir + '/' + 'model_input.conllu', "w", encoding="utf8") as dir:
        for idx, (label, token) in enumerate(zip(labels, tokens)):
            if label:
                row = ["_"] * 10
                row[0] = "1"
                row[1] = token
                row[-1] = str(idx)
                dir.write("\t".join(row))
                dir.write("\n\n")


def read_conllu(tokens: list):
    with open(current_dir + '/' + 'model_output.conllu', "r", encoding="utf8") as src:
        for line in src:
            if len(line) > 5:
                row = line.split('\t')
                idx = int((row[-1].split('\n')[0]))
                tokens[idx] = row[2]
    os.remove(current_dir + '/' + 'model_output.conllu')
    os.remove(current_dir + '/' + 'model_input.conllu')
    return tokens


def correct_errors(tokens, labels):
    make_conllu(tokens, labels)
    corr_command = "python3 {dir}/combo/main.py --mode predict --test {dir}/model_input.conllu "\
                   "--pred {dir}/model_output.conllu --targets lemma --model {dir}/correction_model.pkl".format(dir=current_dir)
    os.system(corr_command)
    corrected_tokens = read_conllu(tokens)
    return corrected_tokens