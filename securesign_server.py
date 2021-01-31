from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import signSGD_Response
from Common.Handler.handler import Handler
from Common.Utils.edcode import encode, decode
import Common.config as config
from Common.Model.LeNet import LeNet
import numpy as np
import torch


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
        self.model = model
        self.root_data = root_data
        self.optimizer = optimizer 
        self.loss_func = loss_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def root_grad_comp(self):
        for X, y in self.root_data:
            
    
    def root_model_update(self):
        pass

    def computation(self, data_in):
        #self.root_grad = root_train(self.root_data, self.model)
        #grad_in = np.array(data_in).reshape((self.num_workers, -1)).sum(axis=0)
        root_grad = root_grad_comp()
        # aggregation
        root_model_update()
        grad_in = np.array(data_in).reshape((self.num_workers, -1))
        print("########")
        for i in range(1,self.num_workers):
            delta = np.logical_xor(grad_in[0], grad_in[i]).astype(int).sum()
            print(delta * 1.0 / grad_in.shape[1])
        grad_in = grad_in.sum(axis=0)
        #grad_out = np.where(grad_in >= (self.num_workers // 2), 1, 0)
        return encode(grad_in.tolist(), BASE=16)


if __name__ == "__main__":
    #cud
    model = torch.load('./Model/LeNet')
    root_data = torch.utils.data.DataLoader('./Data/MNIST/server_data.pt', batch_size=100, shuffle=True, num_workers=0)
    import pdb
    pdb.set_trace()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    gradient_handler = SignSGDGradientHandler(num_workers=config.num_workers, model=model, root_data=root_data, optimizer = opt)

    clear_server = ClearSignSGDServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    clear_server.start()
