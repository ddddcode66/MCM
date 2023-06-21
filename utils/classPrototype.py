import torch
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

def update_sample_feature(args, feature, center):
  for i in range(len(feature)):
    sample = feature[i]
    pos_cos_sim = cos_sim(sample.cpu().detach().numpy().reshape(1,-1), center['pos'].reshape(1,-1))
    neg_cos_sim = cos_sim(sample.cpu().detach().numpy().reshape(1,-1), center['neg'].reshape(1,-1))
    alpha = pos_cos_sim / (pos_cos_sim + neg_cos_sim)
    alpha = torch.tensor(alpha)
    new_sample_add = alpha * torch.tensor(center['pos']) + (1 - alpha) * torch.tensor(center['neg'])
    new_sample_add = new_sample_add.to(args.device)
    new_sample = sample + new_sample_add
    feature[i] = new_sample
  return feature


def update_prototype_feature(args, labels, center_map, feature_map):
    center_map = center_map
    feature_map = feature_map
    def update_single_center(mode):
        neg_indexes = labels < 0
        if args.excludeZero:
            pos_indexes = labels > 0
        else:
            pos_indexes = labels >= 0
        
        if feature_map[mode][pos_indexes].size(0) != 0 and feature_map[mode][neg_indexes].size(0) != 0:
          center_map[mode]['pos'] = GMM(n_components=1).fit(feature_map[mode][pos_indexes].cpu().detach().numpy()).means_
          center_map[mode]['neg'] = GMM(n_components=1).fit(feature_map[mode][neg_indexes].cpu().detach().numpy()).means_
          center_map[mode]['pos'] = torch.tensor(center_map[mode]['pos']).float().to(args.device)
          center_map[mode]['neg'] = torch.tensor(center_map[mode]['neg']).float().to(args.device)
        elif feature_map[mode][pos_indexes].size(0) == 0:
          center_map[mode]['neg'] = GMM(n_components=1).fit(feature_map[mode][neg_indexes].cpu().detach().numpy()).means_
          center_map[mode]['pos'] = torch.zeros(feature_map[mode][0].size(0)).to(args.device)
          center_map[mode]['neg'] = torch.tensor(center_map[mode]['neg']).float().to(args.device)
        else:
          center_map[mode]['pos'] = GMM(n_components=1).fit(feature_map[mode][pos_indexes].cpu().detach().numpy()).means_
          center_map[mode]['pos'] = torch.tensor(center_map[mode]['pos']).float().to(args.device)
          center_map[mode]['neg'] = torch.zeros(feature_map[mode][0].size(0)).to(args.device)
          
    def update_sample_rep(mode):
      for i in range(len(feature_map[mode])):
        sample = feature_map[mode][i]
        pos_cos_sim = cos_sim(sample.cpu().detach().numpy().reshape(1,-1), center_map[mode]['pos'].cpu().detach().numpy().reshape(1,-1))
        neg_cos_sim = cos_sim(sample.cpu().detach().numpy().reshape(1,-1), center_map[mode]['neg'].cpu().detach().numpy().reshape(1,-1))
        alpha = pos_cos_sim / (pos_cos_sim + neg_cos_sim)
        alpha = torch.tensor(alpha).to(args.device)
        new_sample_add = alpha * center_map[mode]['pos'] + (1 - alpha) * center_map[mode]['neg']
        new_sample = sample + new_sample_add
        feature_map[mode][i] = new_sample
    
    update_single_center(mode='text')
    update_single_center(mode='audio')
    update_single_center(mode='video')

    update_sample_rep(mode='text')
    update_sample_rep(mode='audio')
    update_sample_rep(mode='video')

    return feature_map, center_map












