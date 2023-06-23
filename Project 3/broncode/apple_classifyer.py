from torch import nn
import torch.nn.functional
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------------------------------------------------


# Create the convolutional neural net.
class AppleClassifyer(nn.Module):
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, img_size=64, c_in=3, loss_func=nn.CrossEntropyLoss()):
        # Use super() to call the nn.Module.
        super().__init__()

        # Use nn.Sequential() to configure the layers.
        self.model = nn.Sequential(
            nn.Conv2d(c_in, 16, 3, padding=1),
            nn.ReLU(),
            nn. Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn. Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # Flatten for the linear layer.
            nn.Flatten(),
            nn.Linear(img_size*128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            # Use softmax to create an output between 0 and 1 times 4, which makes it so the sum of the 4 outputs equals 1.
            nn.Softmax(dim=1)
            )
        
        self.loss_func = loss_func
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create function to train my model with.
    def fit(self, train_loader, val_loader, test_loader, epochs, lr, opt_function=torch.optim.Adam):
        
        optimizer = opt_function(self.model.parameters(),lr )
        # Set to cuda (gpu)
        self.model.to(self.cuda_available())
        # Create empty list to save the validation results
        history = []
        # Training loop
        for epoch in range(epochs):
            print("epoch:",epoch+1)
            self.model.train()
            
            for batch in train_loader:
                optimizer.zero_grad()
                # I use the function loss_calc that I created below so that I can use this at the validation.
                loss = self.loss_calc(batch)
                loss.backward()
                optimizer.step()

            # Validate
            with torch.no_grad():    
                self.model.eval()
                val_loss = []

                for batch in val_loader:
                    loss = self.loss_calc(batch)
                    val_loss.append(loss)  
            
            # Append the sum(val_loss)
            history.append(sum(val_loss))
            print(sum(val_loss))        
        
        # I use a self made function return a percentage of the accuracy
        acc = round(self.evaluate_accuracy(test_loader))
        print("Accuracy:",acc)
        return history, acc
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # Function for cuda
    def cuda_available(self):

        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create a function to calculate the loss
    def loss_calc(self, batch):
        image = batch[0].to(self.cuda_available())
        labels = batch[1].to(self.cuda_available())
        pred = self(image)        
        loss = self.loss_func(pred, labels)
        return loss

# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create function to evaluate the accuracy of my model
    def evaluate_accuracy(self, test_loader):
        model_pred = []
        actual_pred = []
        cor_pred = 0
        bad_pred = 0
        for batch in test_loader:
            image, labels = batch
            image = image.to(self.cuda_available())
            labels = labels.to(self.cuda_available())
            pred = self(image)
    
            _, y_pred = torch.max(pred,1)                   
            model_pred.extend(y_pred.data.cpu().numpy())
            actual_pred.extend(labels.data.cpu().numpy())

            for image, labels in zip(y_pred, labels):
                if image == labels:
                    cor_pred += 1
                else:
                    bad_pred += 1


        # I decided to add a confusion matrix  
        classes = ('blotch_apples', 'normal_apples', 'rot_apples', 'scab_apples')
        cf_matrix = confusion_matrix(model_pred, actual_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes]).fillna(0)

        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.show()

        acc = cor_pred/(cor_pred + bad_pred) * 100
        return acc
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # A function to predict an image
    def predict_image(self, image):
        image = image.to(self.cuda_available())
        pred = self(image)

        _, y_pred = torch.max(pred,1)                
        result = y_pred

        return result
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    # A function to test a sample and add an AQL label
    def aql_classifyer(self,sample):
        good_apples = 0
        bad_apples = 0
        sample_size = len(sample)

        for image in sample:
            pred = self.predict_image(image)

            if pred == 1:
                good_apples += 1

            else:
                bad_apples += 1

        if bad_apples == 0:
            aql = 0.4
        
        elif 0 <= bad_apples <= 3:
            aql = 6.5

        elif 3<= bad_apples <=15:
            aql = 15

        elif bad_apples > 15:
            aql = "REJECTED"

        statement = f"Amount of good apples: {good_apples}/{sample_size}"
        print(statement)
        return aql, good_apples, bad_apples
# -------------------------------------------------------------------------------------------------------------------------------------------------------
    


    
    def forward(self, data):

        return self.model(data)

