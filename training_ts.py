import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
X = torch.tensor([[1.0, 8.0], [2.0, 3.0], [3.0, 1.0], [4.0, 10.0], [5.0, 11.0],
                  [6.0, 7.0], [7.0, 6.0], [8.0, 10.0], [9.0, 11.0], [10.0, 8.0]])
Y = torch.tensor([[-20.0], [5.0], [3.0], [11.0], [15.0], [11.0], [10.0], [17.0], [50.0], [100.0]])
dataset=TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
class LinearFittingModel(nn.Module):
    def __init__(self):
        super(LinearFittingModel,self).__init__()
        self.linear1 = nn.Linear(2,10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)
        self.linear4 = nn.Linear(10, 10)
        self.linear5 = nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.softplus=nn.Softplus()
        self.elu=nn.ELU()
        self.softsign=nn.Softsign()
        #self.dropout = nn.Dropout(0.1)
    def forward(self,Input):
        x = self.linear1(Input)
        x = self.softsign(x)
        x = self.linear2(x)
        x = self.softsign(x)
        x = self.linear3(x)
        x = self.softsign(x)
        x = self.linear4(x)
        x = self.softsign(x)
        # x =self.dropout(x)
        x = self.linear5(x)
        # x =self.relu(x)
        # x = self.softplus(x)
        #x = self.sigmoid(x)
        return x

model=LinearFittingModel()

criterion=nn.MSELoss()#这一步记住是nn,MSELoss() 其内没有参数
optimizer=torch.optim.SGD(model.parameters(),lr=0.00010,weight_decay=0.,momentum=0.) #这一步记住传入model的可实例化对象 传入学习率

epochs=60000
for epoch in range(epochs):
    for batch_inputs, batch_targets in dataloader:
        outputs=model(batch_inputs)
        loss=criterion(outputs,batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    if loss.item()<0.001:
        break
predicted=model(X).detach().numpy()#detach后不再作为可微分的张量看待 但是仍然是张量 下一步再进行numpy转换 如不detach会报错

fig,ax=plt.subplots()
ax.scatter(X[:,0].numpy(),Y.numpy(),label="Original Data")
ax.plot(X[:,0].numpy(),predicted,label='Fitted Line')
ax.set_title('Plot')
ax.legend()
plt.figure(fig.number)
plt.show()

