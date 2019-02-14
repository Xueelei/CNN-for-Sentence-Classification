from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
from model import Classify
import torch
import torch.optim as optim




class Utils:
    def __init__(self, params, dl):
        self.params = params
        self.data_loader = dl

    @staticmethod
    def to_tensor(arr):
        # list -> Tensor (on GPU if possible)
        if torch.cuda.is_available():
            tensor = torch.tensor(arr).type(torch.cuda.LongTensor)
        else:
            tensor = torch.tensor(arr).type(torch.LongTensor)
        return tensor

    def get_dev_loss_and_acc(self, model, loss_fn):
        losses = []
        hits = 0
        total = 0
        model.eval()
        for sents, lens, labels in self.data_loader.dev_data_loader:
            x_batch = self.to_tensor(sents)
            y_batch = self.to_tensor(labels)

            logits = model(x_batch, lens)
            loss = loss_fn(logits.squeeze(1), y_batch)
            hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            total += len(sents)
            losses.append(loss.item())

        return np.asscalar(np.mean(losses)), hits / total

    def train(self, save_plots_as):
        model = Classify(self.params, vocab_size=len(self.data_loader.w2i),
                         ntags=self.data_loader.ntags)

        
        loss_fn = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        # Variables for plotting
        train_losses = []
        dev_losses = []
        train_accs = []
        dev_accs = []
        s_t = timer()
        prev_best = 0
        patience = 0

        # Start the training loop
        for epoch in tqdm(range(1, self.params.max_epochs + 1)):
            model.train()
            train_loss = 0
            hits = 0
            total = 0
            for sents, lens, labels in self.data_loader.train_data_loader:
                # Converting data to tensors
                #print(sents,labels)
                x_batch = self.to_tensor(sents)
                y_batch = self.to_tensor(labels)
                #print(x_batch,y_batch)

                # Forward pass
                logits = model(x_batch, lens)
                loss = loss_fn(logits.squeeze(1), y_batch)

                # Book keeping
                train_loss += loss.item()
                hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
                # One can alternatively do this accuracy computation on cpu by,
                # moving the logits to cpu: logits.data.cpu().numpy(), and then using numpy argmax.
                # However, we should always avoid moving tensors between devices if possible for faster computation
                total += len(sents)

                # Back-prop
                optimizer.zero_grad()  # Reset the gradients
                loss.backward()  # Back propagate the gradients
                optimizer.step()  # Update the network
            torch.save(model, 'params.pkl')
            
            # Compute loss and acc for dev set
            dev_loss, dev_acc = self.get_dev_loss_and_acc(model, loss_fn)
            train_losses.append(train_loss / len(self.data_loader.train_data_loader))
            dev_losses.append(dev_loss)
            train_accs.append(hits / total)
            dev_accs.append(dev_acc)
            tqdm.write("Epoch: {}, Train loss: {}, Train acc: {}, Dev loss: {}, Dev acc: {}".format(
                        epoch, train_loss, hits / total, dev_loss, dev_acc))
            if dev_acc < prev_best:
                patience += 1
                if patience == 2:
                    # Reduce the learning rate by a factor of 2 if dev acc doesn't increase for 3 epochs
                    # Learning rate annealing
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2
                    optimizer.load_state_dict(optim_state)
                    patience = 0
            else:
                prev_best = dev_acc

        return timer() - s_t
        
    def test(self):
        test_pred = open("result/test_pred.txt","w")
        valid_pred = open("result/valid_pred.txt","w")
        
        model = torch.load('params.pkl')
        i2t = self.data_loader.i2t

        hits = 0
        total = 0
        model.eval()
        
        # Prediction for validation file
        for sents, lens, labels in self.data_loader.dev_data_loader_1:
            x_batch = self.to_tensor(sents)
            y_batch = self.to_tensor(labels)

            logits = model(x_batch, lens)
            
            hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            
            total += len(sents)
            #print(torch.argmax(logits, dim=1), y_batch)
            predict = i2t[int(torch.argmax(logits, dim=1))].capitalize()
            valid_pred.write(predict+"\n")
            

        print(hits/total)

        # Prediction for test file
        for sents, lens, labels in self.data_loader.test_data_loader:
            x_batch = self.to_tensor(sents)

            logits = model(x_batch, lens)

            predict = i2t[int(torch.argmax(logits, dim=1))].capitalize()
            test_pred.write(predict+"\n")


      
        
