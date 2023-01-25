from flask import Flask, render_template, jsonify, request
import os 
import io
import json
import torch
import pandas as pd 
import pickle
from attrdict import AttrDict
from flask import Flask, render_template, request
from cls_model import BertDataset, BertProcessor, BertRegressor, BertTester
from encoder import PolyEncoder, CrossEncoder
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import BertForSequenceClassification
from transform_ver2 import SelectionJoinTransform, SelectionSequentialTransform, SelectionConcatTransform
from chat_function import Poly_function, Cross_function

app = Flask(__name__, static_url_path="/static")   # static_url_path="/static"

def convert_score(score):
    bws_score = int(score) 
    if bws_score > 16: 
        bws_score = 16
    elif bws_score < 0:
        bws_score = 0

    min_score = 0; max_score = 16
    score = (bws_score - min_score) / (max_score - min_score)
    score = int(score * 10)
    return score

def preprocess_sentence(sentence):
    '''
    모델 학습 전에 전처리한 형태로 문장 전처리 
    1. '.' 제거 
    2. "'m -> am"  (Language Detect)
    3. "can't  -> can not"   (Language Detect)
    '''
    input_sent = sentence.replace('.', '')
    input_sent = input_sent.replace("'m", " am")
    input_sent = input_sent.replace("can't", "can not")
    return input_sent

# 감정 분석 페이지
@app.route('/depression_analysis', methods=['GET', 'POST'])
def detectEmotion():
    user_input = ''
    minmax_score = 9999
    emotions = 'ordinary'
    if request.method == 'POST':
        user_input = request.form['user_text']
        model_input = preprocess_sentence(user_input)
        # print(model_input)
        test_data = test_processor.convert_sentence(model_input)
        test_sampler = test_processor.shuffle_data(test_data, 'test')
        test_loader = test_processor.load_data(test_data, test_sampler)
        bws_score, _ = bws_tester.get_score(test_loader, 1)
        dsm_label, _ = dsm_tester.get_label(test_loader, 1)
        emotions = label[dsm_label[0]]
        minmax_score = convert_score(bws_score[0])
        # print(minmax_score, dsm_label)
    return render_template('depression_analysis.html', user_input=user_input, minmax_score=minmax_score, emotions=emotions)


# 챗봇 페이지
@app.route("/chatbot")
def ChatBot():
    return render_template("chatbot.html")

# 사용자 입력 발화 받은 후 답변 생성
@app.route('/chatbot/test', methods=['POST'])
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
    print(jsonify(text=best_response))
    return jsonify(text=best_response)

# 메인 화면
@app.route("/")
def main():
    return render_template("main.html")

if __name__ == '__main__':
    model_path = './model'
    config_path = './config'
    PATH_P = "./model/poly_64_pytorch_model.pt"
    PATH_C = "./model/cross_0_pytorch_model.pt"
    
    global label 
    global test_processor
    global bws_tester
    global dsm_tester 

    with open(os.path.join(config_path, 'training_config.json')) as f:
        training_config = AttrDict(json.load(f))
    training_config.pad = 'max_length'

    label = dict()
    label[0] = 'depressed'   # 우울
    label[1] = 'lethargic'   # 무기력
    label[2] = 'apetite/weight problem'   # 급격한 체중(식욕)변화
    label[3] = 'sleep disorder'   # 수면장애
    label[4] = 'pyschomotor agitaion'   # 정서불안
    label[5] = 'fatigued'   # 피로 
    label[6] = 'guilt and worthless'   # 과도한 죄책감 및 무가치함 
    label[7] = 'cognitive decline'   # 인지기능저하
    label[8] = 'suicidal'   # 자살충동 
    label[9] = 'ordinary'   # 일상 

    bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, 'bert-mini'), model_max_length=32)
    bws_config = BertConfig.from_pretrained(os.path.join(model_path, 'bert-mini', 'bert_config.json'))
    dsm_config = BertConfig.from_pretrained(os.path.join(model_path, 'bert-mini', 'bert_config.json'), num_labels=10)
    bws_model = BertModel.from_pretrained(os.path.join(model_path, 'bert-mini'), config=bws_config)
    dsm_model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, 'bert-mini'), config=dsm_config)
    
    bws_config.max_position_embeddings = 32
    dsm_config.max_position_embeddings = 32
    
    bws_reg = BertRegressor(bws_config, bws_model)
    
    bws_model_name = os.path.join(model_path, 'bert_bws_mini.pt')
    dsm_model_name = os.path.join(model_path, 'bert_mini_5.pt')
    bws_reg.load_state_dict(torch.load(bws_model_name, map_location=torch.device('cpu')))
    dsm_model.load_state_dict(torch.load(dsm_model_name, map_location=torch.device('cpu')))
    
    test_processor = BertProcessor(bws_config, training_config, bert_tokenizer)
    bws_tester = BertTester(training_config, bws_reg)
    dsm_tester = BertTester(training_config, dsm_model)

    # 챗봇 
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
    app.run(debug=True)

if __name__ == '__main__':
    main()