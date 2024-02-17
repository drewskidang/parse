# README for Sean

This guide is designed to help you set up and run the project using WSL (Windows Subsystem for Linux). Please follow the steps below carefully.

## Prerequisites

- Ensure you have Docker installed and running on your system.
- WSL (Windows Subsystem for Linux) should be enabled and set up on your Windows machine.

## Steps to Run

### 1. Ingestion Service Setup

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
python -m vllm.entrypoints.openai.api_server --model "Drewskidang/Textbook_AWQ_DARKSTAR" --quantization awq --enforce-eager --chat-template=chatml --gpu-memory-utilization .7
```

### 4. Running the Chatbot

Finally, to run the chatbot application, use the following command:

```bash
streamlit run app.py --port 8081
```

This will start the Streamlit application on port 8081.

## Conclusion

By following these steps, you should have the project up and running. If you encounter any issues, please review the steps to ensure all commands were executed correctly.