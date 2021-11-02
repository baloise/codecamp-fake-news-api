import numpy as np
import torch
import torch.nn as nn
from flask import Flask, json, request
from transformers import AutoModel, BertTokenizerFast

path = "saved_weights.pt"
bert = None
model = None
tokenizer = None

class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x

# INIT
bert = AutoModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load(path))
model.eval()
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
app = Flask(__name__)

def predict(title):
    tokens_test = tokenizer.batch_encode_plus(
        [title],
        max_length = 15,
        pad_to_max_length=True,
        truncation=True
    )

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    #test_y = torch.tensor(['title', 'label'])

    # deactivate autograd
    with torch.no_grad():
        # model predictions
        
        # sent_id = tokenizer.batch_encode_plus(text, padding=True)

        device = torch.device("cpu")
        preds = model(test_seq.to(device), test_mask.to(device))
        fnstate = np.argmax(preds, axis = 1).detach().cpu().numpy()[0]
        prob = nn.functional.softmax(preds, dim=-1).detach().cpu().numpy()[0]
        isFakeNews = False
        probability = 0
        if (fnstate == 1):
            isFakeNews = True
            probability = prob[1]
        else: 
            probability = prob[0]
        
        return {"isFakeNews": isFakeNews, "probability": str(probability)}

# ROUTE
@app.route("/")
def hello():
    title = str(request.args.get('title'))
    prediction = predict(title)
    response = app.response_class(
        response=json.dumps(prediction),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=False, host='0.0.0.0')




