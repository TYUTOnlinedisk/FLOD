from Common.Node.workerbase import WorkerBase
from Common.Grpc.fl_grpc_pb2 import GradRequest_float
import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet
from Common.Utils.data_loader import load_data_fashion_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub

import numpy as np 


class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer)
        self.client_id = client_id
        self.grad_stub = grad_stub

    def update(self):
        gradients = super().get_gradients()

        res_grad_upd = self.grad_stub.UpdateGrad_float(GradRequest_float(id=self.client_id, grad_ori=gradients))

        super().set_gradients(gradients=res_grad_upd.grad_upd)


if __name__ == '__main__':

    args = args_parser()

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    model = ResNet(BasicBlock, [3, 3, 3])
    train_iter, test_iter = load_data_cifar10(batch_size=batch_size, root='Data/CIFAR10')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("connect success!")

        grad_stub = FL_GrpcStub(grad_channel)

        client = ClearDenseClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_iter,
                                  test_iter=test_iter, config=config, optimizer=optimizer, grad_stub=grad_stub)

        client.fl_train(times=args.E)
        client.write_acc_record(fpath="Eva/clear_avg_acc_cifar10.txt", info="clear_avg_acc_worker_cifar10")
