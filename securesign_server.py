from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import signSGD_Response
from Common.Handler.handler import Handler
from Common.Utils.edcode import encode, decode
import Common.config as config

import numpy as np


class ClearSignSGDServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearSignSGDServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler


    def Update_SignSGD(self, request, context):
        data_dict = {request.id: decode(request.sgn_ori)}
        print("have received:", data_dict.keys())
        rst = super().process(dict_data=data_dict, handler=self.handler.computation)
        return signSGD_Response(sgn_upd=rst)


class SignSGDGradientHandler(Handler):
    def __init__(self, num_workers, model, root_data, lr):
        super(SignSGDGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.model = model
        self.root_data = root_data
        self.lr = lr 
    
    def root_train(root_data, model):
        pass

    def computation(self, data_in):
        self.root_grad = root_train(self.root_data, self.model)
        grad_in = np.array(data_in).reshape((self.num_workers, -1)).sum(axis=0)
        grad_out = np.where(grad_in >= (self.num_workers // 2), 1, 0)
        return encode(grad_out.tolist())


if __name__ == "__main__":
    # cuda set seed
    LeNet = LeNet()
    root_data = # load root data
    gradient_handler = SignSGDGradientHandler(num_workers=config.num_workers, model=LeNet, root_data=, lr=0.001)

    clear_server = ClearSignSGDServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    clear_server.start()
