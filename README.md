# Awesome of Federated Learning

## Developments
### Grpc defined
~~~
rpc UpdateIdx_uint32(IdxRequest_uint32) returns (IdxResponse_uint32){}
rpc UpdateGrad_int32(GradRequest_int32) returns (GradResponse_int32){}
rpc UpdateGrad_float(GradRequest_float) returns (GradResponse_float){}
rpc DataTrans_int32(DataRequest_int32) returns (DataResponse_int32){}
~~~

### Server
1. FlGrpcServer class:
~~~
# collect data
def process(self, dict_data, handler)
# start server
def start(self)
~~~
2. Define your own Server, inherit FlGrpcServer, implement the functions define in Grpc
~~~
from Common.Server.fl_grpc_server import FlGrpcServer

class yourServer(FLGrpcServer):
    # implement your func.
    def yourfunc(self, request, context):
        pass
~~~

3. implememt your hander
~~~
# define interface
class YourHandler:
    def computation(self, data_in):
        return data_out
~~~

### Worker
1. WorkerBase: one pass train
2. Define your Woker，inherit WorkerBase
~~~
from Common.Node.workerbase import WorkerBase

class YourWorker(WorkerBase):
    def update():
        gradients = super().get_gradients()
        
        # define upload / download fucntion
        
        super().set_gradients()    
    
~~~