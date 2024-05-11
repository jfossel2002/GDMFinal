import grpc
import UNKBOT_pb2
import UNKBOT_pb2_grpc


def run_prompt_bot_client(prompt):
    # Open a channel to the server
    with grpc.insecure_channel("localhost:50051") as channel:
        # Create a stub using the generated gRPC class
        stub = UNKBOT_pb2_grpc.UNKServiceStub(channel)
        # Create a request object
        request = UNKBOT_pb2.UNKRequest(prompt=prompt)
        # Make the Prompt_Bot RPC call and get the response
        response = stub.Prompt_Bot(request)
        print("UNK_Bot Response: " + response.response)


def run_train_bot_client(prompt):
    # Open a channel to the server
    with grpc.insecure_channel("localhost:50051") as channel:
        # Create a stub using the generated gRPC class
        stub = UNKBOT_pb2_grpc.UNKServiceStub(channel)
        # Create a request object
        request = UNKBOT_pb2.TrainRequest(prompt=prompt)
        # Make the Train_Bot RPC call and get the response
        response = stub.Train_Bot(request)
        print("Train_Bot Response: " + response.response)


if __name__ == "__main__":
    prompt_bot_example = "Action movie"
   # train_bot_example = "Horror Movie"

    run_prompt_bot_client(prompt_bot_example)
    #run_train_bot_client(train_bot_example)
