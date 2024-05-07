import gc
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from fairtorch import ConstraintLoss, DemographicParityLoss, EqualiedOddsLoss
import dit
from torch.utils.data import DataLoader



#Compute joint distribution P(Z,Y) for binary case.
def P_ZY(Z,Y):
        assert Z.shape==Y.shape
        Z=np.asarray(Z).flatten()
        Y=np.asarray(Y).flatten()
        H, xedges, yedges= np.histogram2d(Z, Y, bins=2)
        H=H/Z.shape[0]
        p00=H[0,0]
        p01=H[0,1]
        p10=H[1,0]
        p11=H[1,1]

        return p00, p01, p10, p11


#Given joing distribution B, function computes PID terms. 
def PID(A,B):
    #A=['000','001','010','011','100','101','110','111']
    fd=dit.Distribution(A,B)
    fd.set_rv_names('SZY')
    fd_pid = dit.pid.PID_BROJA(fd, ['S', 'Y'], 'Z')
    Uniq=fd_pid._pis[(('Y',),)]
    Red = fd_pid._pis[(('S',), ('Y',))]
    Syn = fd_pid._pis[(('S', 'Y'),)]
    Uniq_S = fd_pid._pis[(('S',),)]
    I_YS=dit.shannon.mutual_information(fd, ['Y'], ['S'])
    I_ZS=dit.shannon.mutual_information(fd, ['Z'], ['S'])
    I_ZY=dit.shannon.mutual_information(fd, ['Z'], ['Y'])
    I_ZY_S= dit.multivariate.coinformation(fd, 'ZY', 'S')
    CoI= dit.multivariate.coinformation(fd, ['Y', 'Z', 'S'])

    return Uniq, Red, Syn ,Uniq_S, I_ZS, I_YS, I_ZY, I_ZY_S , CoI



logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]


    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels, sensitive_features in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(data)
                outputs = outputs.to(torch.float32)
                labels = labels.to(torch.float32)

                dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100)
                fainess_lamdba = 2
           
                loss=eval(self.criterion)()(outputs, labels) + fainess_lamdba*dp_loss(data, outputs, sensitive_features)
                loss.backward()
                optimizer.step() 

                if self.device == "cuda": torch.cuda.empty_cache()               
        self.model.to("cpu")
    
    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels, sensitive_features in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                outputs = outputs.to(torch.float32)
                labels = labels.to(torch.float32)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                predicted = outputs.round()
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        
        with torch.no_grad():
            outputs_me = self.model(self.data.tensors[0])
            labels_me = self.data.tensors[1]
            predicted_me = outputs_me.round()
               
            acc = predicted_me.eq(labels_me.view_as(predicted_me)).sum().item()/len(self.data.tensors[1])
            self.zy= P_ZY(self.data.tensors[2],predicted_me)
            p00, p01, p10, p11 = self.zy

            SP= p11-p01 #local statistical parity 

            As=['00','01','10','11']
            dists=dit.Distribution(As,self.zy)
            dists.set_rv_names('ZY')
            I_ZYs=dit.shannon.mutual_information(dists, ['Z'], ['Y'])

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> I(Z,Y| S= {self.id}) : {I_ZYs:.4f}\
            \n\t=> SP: {SP:.4f}\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return test_loss, test_accuracy
