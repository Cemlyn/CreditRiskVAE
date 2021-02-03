import torch
import pandas as pd, numpy as np
from Vaemodel import VAE
from torch.utils.data import DataLoader
from torch import optim
import time
from io import StringIO

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

def prep_data():
    df = pd.read_csv('./data/vae_train.csv').sample(10**4)
    for col in df:
        df[col] = df[col].astype(np.float32)
    
    X,y_actual = df.drop(columns=['TARGET']).values,df['TARGET'].values
    X = torch.from_numpy(X)
    y_actual = torch.from_numpy(y_actual)

    dataloader = DataLoader(X, batch_size=300, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)


    return X,y_actual


if __name__=='__main__':
    #run()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Cuda Device Available")
        print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        print("GPU Computational Capablity: ", torch.cuda.get_device_capability())

    model = VAE(latent_dim=2)
    if torch.cuda.is_available():
        model.cuda()
    
    dataloader = prep_data()
    X,y_actual = prep_data()

    if torch.cuda.is_available():
        X,y_actual = X.to('cuda'),y_actual.to('cuda')
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    t0 = time.time()
    for i in range(10**5):
        optimizer.zero_grad()
        q, mu, sigma = model(X)
        loss = model.loss_function(q,X,mu,sigma)
        loss.backward()
        optimizer.step()

        # for batch_id, batch_sample in enumerate(dataloader):
        #     optimizer.zero_grad()
        #     q, mu, sigma = model(batch_sample)
        #     loss = model.loss_function(q,X,mu,sigma)
        #     loss.backward()
        #     optimizer.step()
        
        t1 = time.time()
        t = t1-t0

        if (i%1000)==0:
            string = f'Epoch:{i}, Loss:{loss}, t:{t:0.2f}'
            print(string)
            with open('performance.csv','a') as file:
                file.write(string)
                file.write('\n')