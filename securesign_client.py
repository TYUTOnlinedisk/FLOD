from Common.Node.workerbase import WorkerBase
from Common.Grpc.fl_grpc_pb2 import signSGD_Request
from Common.Utils.edcode import encode, decode
import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Utils.data_loader import load_data_fashion_mnist
from Common.Utils.set_log import setup_logging

import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub

import argparse


class ClearSignSGDClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, grad_stub):
        super(ClearSignSGDClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer)
        self.client_id = client_id
        self.grad_stub = grad_stub

    def update(self):
        gradients = super().get_gradients()
        sgn = np.where(np.array(gradients)>=0, 0, 1)
        res_sgn_upd = self.grad_stub.Update_SignSGD.future(signSGD_Request(id=self.client_id, sgn_ori=encode(sgn)))
        res_sgn = decode(res_sgn_upd.result().sgn_upd)[:super()._grad_len]
        super().set_gradients(gradients=res_sgn_upd.sgn_upd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='signSGD_client')
    parser.add_argument('-i', type=int, help="client's id")
    parser.add_argument('-t', type=int, default=1, help="train passes locally")

    args = parser.parse_args()

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    model = LeNet()
    if args.id == 0:
        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path)
    else:
        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path), None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("connect success!")

        sgn_stub = FL_GrpcStub(grad_channel)

        client = ClearSignSGDClient(client_id=args.i, model=model, loss_func=loss_func, train_iter=train_iter,
                                  test_iter=test_iter, config=config, optimizer=optimizer, grad_stub=sgn_stub)

        client.fl_train(times=args.t)
        client.write_acc_record(fpath="Eva/clear_signSGD_acc.txt", info="clear_signSGD_acc_worker")
