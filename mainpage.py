from flask import Flask, render_template, jsonify, request
import os 
import json
import torch
from attrdict import AttrDict
from flask import Flask, render_template, request
from cls_model import BertDataset, BertProcessor, BertRegressor, BertTester
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import BertForSequenceClassification

app = Flask(__name__)

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

# 메인 화면
@app.route("/")
def main():
    return render_template("main.html")

# 감정 분석 페이지
@app.route('/depression_analysis', methods=['GET', 'POST'])
def detectEmotion():
    user_input = ''
    minmax_score = 9999
    emotions = ''
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

if __name__ == '__main__':
    model_path = './model'
    config_path = './config'

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

    app.run(debug=True)

if __name__ == '__main__':
    main()