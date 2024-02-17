# README for Sean

This guide is designed to help you set up and run the project because you don't dev enough using WSL (Windows Subsystem for Linux). Please follow the steps below carefully.
Utlizing 
-Llama-index
-VLLM
-Langchain
-Pinecone cuz its easier
-Chainlit for UI 

## Prerequisites

- Ensure you have Docker installed and running on your system.
- WSL (Windows Subsystem for Linux) should be enabled and set up on your Windows machine.
- Create .env file like PINECONE_API_KEY= 

- You can add  `OPENAI_API_KEY=` for embeddings in case you don't have a GPU. Required dimension size is 1536.
- Actually Voyage is better so do VOYAGE_API_KEY=
- https://docs.llamaindex.ai/en/stable/examples/embeddings/voyageai.html 
- https://www.voyageai.com/ 
Please install pip install llama-index-embeddings-voyageai
- When you create a Pinecone index, my model has a 768 dimension size.. https://huggingface.co/Drewskidang/bert_base3. Embeddings is more for Legal. If you fine that the model struggles on general query you can try https://huggingface.co/Drewskidang/FEDBGE or just use voyage 

## Steps to Run

### 1. Ingestion Service Setup

Thanks to https://github.com/nlmatics/nlm-ingestor for open-sourcing this paraser. Best parser I used. 

First, we need to set up the ingestion service using Docker:

```bash
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest
```

This will pull the latest version of the `nlm-ingestor` and run it, mapping the container's port 5001 to your local port 5010.

### 2. Running `ingest.py`

Before running `ingest.py`, ensure you have created the necessary Pinecone index as required by the script.

Open a new terminal and navigate to the project directory. Run the following command:

```bash
python ingest.py
```

### 3. Model Loading

In a separate terminal, you need to export your AI keys and run the model server. Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python -m vllm.entrypoints.openai.api_server --model "Drewskidang/AWG_MERGE" --quantization awq --enforce-eager --chat-template=chatml --gpu-memory-utilization .7
```

### 4. Running the Chatbot

Finally, to run the chatbot application, use the following command:

```bash
chainlit run app.py --port 8081
```

This will start the Chainlit application on port 8081.

## Conclusion

By following these steps, you should have the project up and running. If you encounter any issues, please review the steps to ensure all commands were executed correctly. Also there's no memory, i dont how to config chainlit with llama-index and memory. It doese not working when using pinecone. I cant english 