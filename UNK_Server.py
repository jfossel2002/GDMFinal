from concurrent import futures
import grpc
from grpc_reflection.v1alpha import reflection
import UNKBOT_pb2
import UNKBOT_pb2_grpc
import Mini_GPT

class UNKServiceServicer(UNKBOT_pb2_grpc.UNKServiceServicer):
    def Prompt_Bot(self, request, context):
        bot_response = Mini_GPT.runBot(request.prompt)
        response = UNKBOT_pb2.UNKResponse(response=bot_response)
        return response

    def Train_Bot(self, request, context):
        bot_response = Mini_GPT.startModel(request.prompt)
        response = UNKBOT_pb2.TrainResponse(response=bot_response)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    UNKBOT_pb2_grpc.add_UNKServiceServicer_to_server(UNKServiceServicer(), server)
    
    SERVICE_NAMES = (
        UNKBOT_pb2.DESCRIPTOR.services_by_name['UNKService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()