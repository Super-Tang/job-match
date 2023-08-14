import time

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataloader import *
from model import *

my_processor = DataPreprocessor()
train_data = my_processor.get_train_examples('data.json')

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
label_list = my_processor.get_labels()

train_features = convert_examples_to_features(train_data, label_list, 256, tokenizer)
train_dataset = MyDataset(train_features)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

train_data_len = len(train_dataset)
print(f"训练集长度：{train_data_len}")

device = torch.device('cuda')
my_model = RankNet(device).to(device)
loss_fn = focal_loss
my_model.load_state_dict(torch.load('best_model.pth'), False)
learning_rate = 1e-5
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
epoch = 60
my_model.train()
print(device)
acc = 0


for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")
    start_time = time.time()
    train_total_accuracy = 0
    for step, batch_data in enumerate(train_data_loader):
        output = my_model(**batch_data)
        output = output.squeeze(-1)
        batch_label = batch_data['label_id'].to(device).to(torch.float64)
        loss = loss_fn(output, batch_label)
        result = [1 if o > 0.5 else 0 for o in output]
        train_accuracy = sum([1 if i == j else 0 for i, j in zip(result, batch_label)])
        train_total_accuracy = train_total_accuracy + train_accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_total_accuracy = train_total_accuracy / train_data_len
    if train_total_accuracy > acc:
        acc = train_total_accuracy
        torch.save(my_model.state_dict(), 'best_model1.pth')
    end_time = time.time()
    total_train_time = end_time - start_time
    print(f'训练时间: {total_train_time}秒')
    print(f"训练集上的准确率：{train_total_accuracy}")
    #test_acc = test(my_model, test_data_loader)
    #print(f'测试集上的准确率：{test_acc}')
    #my_model.train()

