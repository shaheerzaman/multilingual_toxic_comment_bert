class BERTDatasetTraining:
    def __init__(self, comment_text, targets, tokenizer, max_length):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets
        
    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = ' '.join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            truncating=True
        )

        ids = inputs['input_ids']
        token_type_ids = inupts['token_type_ids']
        mask = inputs['attention_mask']

        padding_length = self.max_length - len(ids)
        
        ids = ids + ([0]*padding_length)
        mask = mask + ([0]*padding_length)
        token_type_ids = token_type_ids + ([0]*padding_length)

        return {
            'ids':torch.tensor(ids, dtype=torch.long), 
            'mask':torch.tensor(mask, dtype=torch.long), 
            'token_type_ids':torch.tensor(token_type_ids, dtype=torch.long),
            'targets':torch.tensor(self.targets[item], dtype=torch.float)
        }

