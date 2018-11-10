from dataset import MomentsDataset
from torch.utils.data import Dataset, DataLoader


dataset_path = '/hdd/datasets/moments/Moments_in_Time_256x256_30fps/'

ds = MomentsDataset(dataset_path)

dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)

for batch in dataloader:
    print(batch['images'].shape)
    print(batch['label'])


