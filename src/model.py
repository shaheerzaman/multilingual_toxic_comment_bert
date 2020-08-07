import torch
import transfomers

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropuout(0.3)
        self.out = nn.Linear(768*2, 1)
        
    def forward(self, ids, mask, token_type_ids):
        o1, o2 = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )
        apool = torch.mean(o1,  1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)

        bo = self.bert_drop(cat)
        p2 = self.out(bo)
        return p2

