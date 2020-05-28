import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import pandas as pd

import seaborn as sns

sns.set()

from src.dataset_creator import DataSetCreator
from src.batch_gen import BatchGen


class Net(nn.Module):
    def __init__(self):
        channels = 3
        descriptor_size = 16

        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 7, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12 * 12 * 7, 256)
        self.fc2 = nn.Linear(256, descriptor_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def forward(self, anchor, puller, pusher):
        temp = torch.ones([anchor.shape[0]], dtype=torch.float32)
        diff_pos = torch.sum((anchor - puller)**2,1)
        diff_neg = torch.sum((anchor - pusher)**2,1)
       
        diff = temp - (diff_neg/(diff_pos + self.margin*temp))
        loss_t = torch.mean(torch.clamp(diff, min=0.0))
        loss_p = torch.mean(diff_pos)
        loss = loss_t + loss_p
        
        return loss



class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        return embedded_x, embedded_y, embedded_z


def img2input(path, loader):
    x = loader(Image.open(path)).float()
    x = Variable(x, requires_grad=True)
    return x


model = Net()
tnet = TripletNet(model)

num_epochs = 5
learning_rate = 0.0001
batch_size = 32
SAVING_PATH = 'tnet.ckpt'
TRAIN = False

criterion_triplet = nn.TripletMarginLoss(margin=0.1)
criterion_reg = nn.L1Loss(size_average=False)
criterion_pair = nn.PairwiseDistance()

optimizer = torch.optim.Adam(tnet.parameters(), lr=learning_rate)
loader = transforms.Compose([transforms.ToTensor()])

print("BATCH GENERATION...")
dataset = DataSetCreator()
triplet_gen = BatchGen(dataset.get_train_data(), dataset.get_test_data(), dataset.get_db())
triplets = triplet_gen.get_all_triplets()

# convert to tensors
triplet_images = list()
for i in range(len(triplets)):
    a, puller, pusher = img2input(triplets[i][0][0], loader), img2input(triplets[i][1][0], loader), img2input(triplets[i][2][0], loader)
    triplet_images.append([a, puller, pusher])

data_loader = DataLoader(triplet_images, batch_size=batch_size)

if TRAIN:
    print("TRAINING STARTING...")
    for i in range(num_epochs):
        print(f'====== Epoch {i} ======')
        for batch_idx, data in enumerate(data_loader):
            o1, o2, o3 = tnet(data[0], data[1], data[2])
            loss_trip = criterion_triplet(o1, o2, o3)
            loss_pair = criterion_pair(o1, o2)
            loss_reg = Variable(torch.FloatTensor(1), requires_grad=True)
            for param in tnet.parameters():
                loss_reg = loss_reg + param.norm(2)
            total_loss = loss_trip + loss_pair + 0.0001 * loss_reg
            optimizer.zero_grad()
            total_loss.sum().backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch {i} Loss: {total_loss.sum().item()}')
    torch.save(tnet.state_dict(), SAVING_PATH)

model = Net()
tnet = TripletNet(model)
tnet.load_state_dict(torch.load(SAVING_PATH))
tnet.eval()

print("CALCULATING DESCRIPTORS FOR DB")
db = dataset.get_db()
db_ds = list()
for obj in tqdm(list(db.keys())):
    for i in range(len(db[obj])):
        pred = tnet.embeddingnet(img2input(db[obj][i][0], loader).unsqueeze(0))
        db_ds.append([obj, db[obj][i][0], db[obj][i][1], pred[0].detach().numpy()])

print("CALCULATING DESCRIPTORS FOR TESTSET")
test_ds = list()
test = dataset.get_test_data()
for obj in tqdm(list(test.keys())):
    for i in range(len(test[obj])):
        pred = tnet.embeddingnet(img2input(test[obj][i][0], loader).unsqueeze(0))
        test_ds.append([obj, test[obj][i][0], test[obj][i][1], pred[0].detach().numpy()])

print("PERFORM KNN")
db_ds = np.array(db_ds).transpose()
test_ds = np.array(test_ds).transpose()

knn = KNeighborsClassifier(n_neighbors=1)
indices = list(range(db_ds.shape[1]))
knn.fit(np.stack(db_ds[3]), indices)
final_pred = knn.predict(np.stack(test_ds[3]))

angualar_differences = list()
cnt = 0
predicted = list()
for i in range(len(test_ds[0])):
    predicted.append(db_ds[0][final_pred[i]])
    if test_ds[0][i] == db_ds[0][final_pred[i]]:
        cnt += 1
        diff = triplet_gen.quat_angular_metric(test_ds[2][i], db_ds[2][final_pred[i]])
        angualar_differences.append(diff)

print(f'{cnt}/{len(final_pred)} are correctly classified')

# Plot histogram
n = len(final_pred)
bins = [0, 0, 0, 0, 0, 0, 0]
angles = [0.5, 1, 5, 10, 20, 40, 180]

for a in angualar_differences:
    for i in range(len(angles)):
        if a < angles[i]:
            bins[i] += 1

y_pos = np.arange(len(angles))
bins = [(x / n) * 100 for x in bins]
plt.bar(y_pos, bins)
plt.xticks(y_pos, angles)
plt.savefig("hist.png", dpi=300)
plt.show()

print("COMPUTING TSNE...")
X_embedded = TSNE(n_components=3).fit_transform(np.stack(test_ds[3]))
X_embedded = X_embedded.transpose()
print(X_embedded.shape)

# Plot t-SNE
df = pd.DataFrame(test_ds[0], columns=['data'])
df['data'] = pd.Categorical(df['data'])
my_color = df['data'].cat.codes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_embedded[0], X_embedded[1], X_embedded[2], c=my_color, cmap="viridis")
plt.savefig("tsne.png", dpi=300)
plt.show()

# Plot confusion matrix
objects = list(db.keys())
conf_matrix = confusion_matrix(test_ds[0], predicted, labels=objects)
conf_matrix = [[item / n * 100 for item in line] for line in conf_matrix]

ax = sns.heatmap(conf_matrix, cmap="Blues", xticklabels=objects, yticklabels=objects)
ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right')
ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion.png", dpi=300)
plt.show()
