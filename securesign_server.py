from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import signSGD_Response
from Common.Handler.handler import Handler
from Common.Utils.edcode import encode, decode
import Common.config as config
from Common.Model.LeNet import LeNet
import numpy as np
import torch
from Common.Utils.options import args_parser

class ClearSignSGDServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearSignSGDServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

    def Update_SignSGD(self, request, context):
        data_dict = {request.id: decode(request.sgn_ori, BASE=2)}
        print("have received:", data_dict.keys())
        rst = super().process(dict_data=data_dict, handler=self.handler.computation)
        return signSGD_Response(sgn_upd=rst)


class SignSGDGradientHandler(Handler):
    def __init__(self, num_workers, model, root_data, optimizer, loss_func):
        super(SignSGDGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.root_data = root_data
        self.optimizer = optimizer 
        self.loss_func = loss_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self._gradients = None
        self._level_length = None
        self._grad_len = 0
    
    def train_step(self, x, y):
        """ Find the update gradient of each step in collaborative learning """
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()

        self._gradients = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.grad.numel() + self._level_length[-1])
            self._gradients += param.grad.view(-1).numpy().tolist()

        self._grad_len = len(self._gradients)

    def root_grad_comp(self):
        for X, y in self.root_data:
            self.train_step(X, y)
        root_sgn = np.where(np.array(self._gradients)>=0, 0, 1)
        self._gradients = None
        return root_sgn

    def root_model_update(self, grad):
        idx = 0
        for param in self.model.parameters():
            tmp = self._gradients[self._level_length[idx]:self._level_length[idx + 1]]
            grad_re = torch.tensor(tmp, device=self.device)
            grad_re = grad_re.view(param.grad.size())

            param.grad = grad_re
            idx += 1
        self.optimizer.step()

    def computation(self, data_in):
        #self.root_grad = root_train(self.root_data, self.model)
        grad_in = np.array(data_in).reshape((self.num_workers, -1))
        root_sgn = self.root_grad_comp()
        assert grad_in.shape[1] = root_sgn.shape[0]
        T = []
        for i in range(self.num_workers):
            hamming_distance = np.logical_xor(root_sgn, grad_in[i]).astype(int).sum(axis=0)
            if hamming_distance > (self._grad_len // 2):
                T[i] = 0
            else:
                T[i] = (self._grad_len // 2 - hamming_distance) / (self._grad_len // 2)
        scaler = (1.0 / sum(T[i])) 
        weight_sgn = 0
        for i in range(self.num_workers):
            weight_sgn += T[i] * grad_in[i]
        grad_agg = scaler * weight_sgn  
        self.root_model_update(grad=grad_agg)
        #grad_out = np.where(grad_in >= (self.num_workers // 2), 1, 0)
        return encode(grad_agg, BASE=16)


if __name__ == "__main__":
    args = args_parser()
    model = torch.load('./Model/LeNet')
    loss_func = nn.CrossEntropyLoss()
    data_path = torch.load('./Data/MNIST/server_data.pt')
    root_data = torch.utils.data.DataLoader(data_path, batch_size=100, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=agrs.lr)
    gradient_handler = SignSGDGradientHandler(num_workers=config.num_workers, model=model, root_data=root_data, optimizer = opt, loss_func=loss_func)

    clear_server = ClearSignSGDServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    clear_server.start()
