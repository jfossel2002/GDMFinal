//DEPRECATED - This server doesn't work
const express = require('express');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');
const cors = require('cors');

const PROTO_PATH = path.join(__dirname, 'UNKBOT.proto');
const app = express();

app.use(cors());
app.use(express.json());

const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true
});

const UNKBOTProto = grpc.loadPackageDefinition(packageDefinition).UNKBOT;

if (!UNKBOTProto || !UNKBOTProto.UNKService) {
    console.error("Failed to load UNKService from proto file");
    process.exit(1);
}

const client = new UNKBOTProto.UNKService('localhost:50051', grpc.credentials.createInsecure());

app.post('/prompt', (req, res) => {
    const { prompt } = req.body;
    console.log('Received prompt:', prompt);

    const request = { prompt: prompt };
    client.Prompt_Bot(request, (error, response) => {
        if (error) {
            console.error('gRPC Error:', error);
            res.status(500).send(error.message);
        } else {
            console.log('gRPC Response:', response);
            res.json(response);
        }
    });
});

app.listen(4000, () => {
    console.log('Express server listening on port 4000');
});
