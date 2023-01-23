from flask import Flask, render_template, request
from flask import jsonify

# libraries
import pickle
import io
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from encoder import PolyEncoder, CrossEncoder
from transform_ver2 import SelectionJoinTransform, SelectionSequentialTransform, SelectionConcatTransform
from chat_function import Poly_function, Cross_function

# cpu 이용시 pickle 파일 업로드
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# 데이터 불러오기
data = pd.read_csv("./data/candidate.csv", encoding = 'utf8', index_col = 0)
data = data.reset_index(drop=True)
cand_data = data['response']

# 사전 계산된 후보군 tensor 값 불러오기

# gpu
# with open("./data/candidate_emb.pickle", "rb") as file:
# final_cand_emb = pickle.load(file)

# cpu
with open("./data/candidate_emb.pickle", "rb") as file:
    final_cand_emb = CPU_Unpickler(file).load()

# gpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#cpu
device = torch.device("cpu")

# Poly Encoder 사전 학습 BERT 모델
bert_name = 'google/bert_uncased_L-4_H-512_A-8'
bert_config = BertConfig.from_pretrained(bert_name)
bert = BertModel.from_pretrained(bert_name, config=bert_config)

# Cross Encoder 사전 학습 BERT 모델
rerank_name = 'google/bert_uncased_L-4_H-512_A-8'
rerank_config = BertConfig.from_pretrained(rerank_name)
rerank_bert = BertModel.from_pretrained(rerank_name, config=bert_config)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_name)
tokenizer.add_tokens(['\n'], special_tokens = True)

# reranker 모델 Tokenizer
rerank_tokenizer = BertTokenizer.from_pretrained(bert_name)
rerank_tokenizer.add_tokens(['\n'], special_tokens=True)

# 모델 경로 설정
PATH_P = "./model/poly_64_pytorch_model.pt"
PATH_C = "./model/cross_0_pytorch_model.pt"

# Poly Encoder 모델 불러오기
model = PolyEncoder(bert_config, bert=bert, poly_m=64)
model.resize_token_embeddings(len(tokenizer))
# gpu
# model.load_state_dict(torch.load(PATH_P))
# cpu
model.load_state_dict(torch.load(PATH_P, map_location=device))
model.to(device)
model.device

# Cross Encoder 모델 불러오기
rerank_model = CrossEncoder(rerank_config, bert=rerank_bert)
rerank_model.resize_token_embeddings(len(rerank_tokenizer))
# gpu
# rerank_model.load_state_dict(torch.load(PATH_C))
# cpu
rerank_model.load_state_dict(torch.load(PATH_C, map_location=device))
rerank_model.to(device)
rerank_model.device


app = Flask(__name__,static_url_path="/static")

# 사용자 입력 발화 받은 후 답변 생성
@app.route('/test', methods=['POST'])
def reply():

    text = request.json['msg']
    user_context = [str(text)]
    
    idx_list = []
    top_cand = []
    
    # 인코더별 입력 발화 토큰화
    # 모델에 따라 max_len 길이 달라질 수 있음
    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=512)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=40)
    concat_transform = SelectionConcatTransform(tokenizer=rerank_tokenizer, max_len=512)
    
    # 인코더별 함수 정의(연산과정)
    p_func = Poly_function(model=model, bert=bert, tokenizer=tokenizer, device = device, 
                           context_transform = context_transform, response_transform = response_transform)
    c_func = Cross_function(model=rerank_model, bert=rerank_bert, tokenizer=rerank_tokenizer, device = device,
                            concat_transform = concat_transform)
    
    # top5 후보군 추출 과정
    # gpu 이용시 후보군 갯수 늘릴 수 있음
    user_emb = p_func.ctx_emb(*p_func.input_context(user_context))
    final_score = p_func.score(user_emb, final_cand_emb)
    new_score = final_score.sort()
    for i in range(5):
        idx_list.append(int(new_score[1][0][-5:][i]))
    for idx in reversed(idx_list):
        top_cand.append(cand_data[idx])

    # top5
    top_cand = pd.Series(top_cand)

    # 최종 답변 추출 과정 - 재순위화
    rerank_list = []
    
    for i in range(len(top_cand)):
        response = [top_cand[i]]
        cross_score = c_func.text_emb(*c_func.input_text(user_context, response))
        rerank_list.append(cross_score.item())
    
    
    # 최종 답변
    final_index = rerank_list.index(max(rerank_list))
    best_response = top_cand[final_index]

    return jsonify(text=best_response)

@app.route("/")
def chat():
    return render_template("chat_ver1.html")

if __name__ == '__main__':
    app.run()