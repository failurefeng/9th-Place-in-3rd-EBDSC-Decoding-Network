import torch 

pretrain = torch.load('./results/Transformer_puredemod/best/ours/checkpoints/cosscheduler/best_CQ_epoch26.pth')
print('epoch:', pretrain['epoch'], 'best_CQ_score:',pretrain['best_CQ_score'])

