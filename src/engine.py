import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        targets = d['targets']
        
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs, targets)
        if bi%10 == 0:
            xm.master_print(f'bi={bi}, loss={loss}')
        
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            ids=ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )
        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()

        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)

    return fin_outputs, fin_targets

        
