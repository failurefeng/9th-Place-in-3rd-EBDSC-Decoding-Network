import torch 

pretrain = torch.load('./checkpoints/best_CQ_old1.pth')
print('epoch:', pretrain['epoch'], 'best_CQ_score:',pretrain['best_CQ_score'])
print('state_dict:', pretrain['state_dict'].keys())

