import json

import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, TFBertModel

from utils import load_dataset

df = load_dataset("../../data/insurance_qna_dataset.csv")
questions = np.unique(df.iloc[:, 0].to_numpy())

# tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
# model = TFBertModel.from_pretrained("ProsusAI/finbert")

# Transform and save input questions
# outputs = []
# for i, q in enumerate(questions):
#     encoded_text = tokenizer(q, return_tensors='tf')
#     output = model(encoded_text)[1].numpy().reshape(-1)
#     outputs.append(output)
#     if i % 100 == 0:
#         print(i)

# outputs = np.asarray(outputs)
# np.save('finbert_emmbedings.npy', outputs)
outputs = np.load("finbert_emmbedings.npy")

knn = NearestNeighbors(n_neighbors=100, metric="cosine").fit(outputs)

with open("../../data/test_questions_json.json") as json_file:
    json_data = json.load(json_file)

test_questions = json_data["question"]
original = json_data["original"]

# Transform and save test questions
# tq = []
# for i, q in enumerate(test_questions):
#     encoded_text = tokenizer(q, return_tensors='tf')
#     output = model(encoded_text)[1].numpy().reshape(-1)
#     tq.append(output)
#
# tq = np.asarray(tq)
# np.save('finbert_emmbedings_test.npy', tq)
tq = np.load("finbert_emmbedings_test.npy")

_, indices = knn.kneighbors(tq)

indices_original = np.asarray([questions.tolist().index(o) for o in original])

rank = np.where(indices == indices_original[:, None])[1]
penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
score = (rank.sum() + penalization) / indices_original.shape[0]

print(f"Score: {score}")
