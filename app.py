import os
import json

from llama_index.core.response import Response 

from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core import StorageContext
from langchain_community.llms import VLLMOpenAI
import chainlit as cl
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core.query_engine import CitationQueryEngine
embed_model = HuggingFaceEmbedding(model_name="Drewskidang/ANTI_BERT")

from transformers import AutoTokenizer
Settings.tokenzier = AutoTokenizer.from_pretrained(
    "Drewskidang/Textbook_AWQ_DARKSTAR"
)
api_key = os.environ.get("PINECONE_API_KEY","")
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("legalbert")

pc = Pinecone(api_key=api_key)

from langchain_openai import ChatOpenAI

embed_model =embed_model

Settings.embed_model =embed_model


# Iterate over all PDF files in the directory
inference_server_url="http://localhost:8000/v1"
llm=VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        model_name="Drewskidang/Textbook_AWQ_DARKSTAR",
        max_tokens=512
    )
Settings.llm=llm


#EXPERT_PROMPT = PromptTemplate(expert_prompt_tmpl_str)
inference_server_url="http://localhost:8000/v1"
STREAMING = True
@cl.on_chat_start
async def factory():
    llm=VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        model_name="Drewskidang/Textbook_AWQ_DARKSTAR",
        max_tokens=512
    )
 

    pinecone_index = pc.Index("legalbert")
    name_space="antitrust"
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(storage_context=storage_context,name_space=name_space,vector_store=vector_store)
    query_engine = index.as_query_engine(llm=llm,vector_store_query_mode="hybrid")
    #query_engine.update_prompts(
    #{"response_synthesizer:text_qa_template": EXPERT_PROMPT})
    cl.user_session.set("query_engine", query_engine)
    
    
    
@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")
    await response_message.send()

    if isinstance(response, Response):
        response_message.content = str(response)
        await response_message.update()
    elif isinstance(response, Response):
        gen = response.response_gen
        for token in gen:
            await response_message.stream_token(token=token)

        if response.response_txt:
            response_message.content = response.response_txt
            await response_message.send()
    label_list = []
    count = 1
    for sr in response.source_nodes:
            elements = [cl.Text(name="S"+str(count), content=f"{sr.node.text}", display="side", size='small')]
            response_message.elements = elements
            label_list.append("S"+str(count))
            await response_message.update()
            count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)

    await response_message.update()    

