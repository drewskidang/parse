import os
import json
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import LayoutReader
from llama_index.core import Document
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core import StorageContext
from langchain_openai  import ChatOpenAI
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.util import assemble
import logging
import spacy_llm
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.DEBUG)
nlp = assemble("config.cfg")

api_key = os.environ.get("PINECONE_API_KEY","")
pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("replace_with_index")
model_name = "voyage-02"
voyage_api_key = os.environ.get("VOYAGE_API_KEY", "")

embed_model = HuggingFaceEmbedding(model_name="Drewskidang/ANTI_BERT")

def get_extra_info():
    print("\nPlease enter the following information for each document:")
    print("Book/Case information, Relevant Rules, Case Name, Related Cases")
    law_subject_area = input("Law subject area: ")
    relevant_rules = input("Relevant Rules: ")
    case_name = input("Case Name: ")
    related_cases = input("Related Cases: ")
    
    return {
        "law_subject_area": law_subject_area,
        "relevant_rules": relevant_rules,
        "extra_data": case_name,
        "related_cases": related_cases
    }
inference_server_url="http://localhost:8000/v1"

llm=ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        model_name="Drewskidang/Marlin-AWQ",
        max_tokens=512
    )


Settings.llm=llm
Settings.embed_model=embed_model



def process_pdfs(pdf_directory):
    parser_api_url  = "http://localhost:5010/api/parseDocument?renderFormat=all&applyOcr=yes"
    pdf_reader = LayoutPDFReader(parser_api_url)

    data = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"\nProcessing document: {filename}")  # Display the name of the document
            docs = pdf_reader.read_pdf(pdf_path)    # Move the call to get_extra_info() here, so it's called once per document
            extra_info = get_extra_info()  # Get extra info from the user for each document
            for chunk  in docs.chunks(): 
                chunk_text = chunk.to_context_text()
                docs= nlp(chunk_text)
                document = Document(
                            text=chunk.to_text(include_children=True, recurse=True),
                            extra_info=extra_info,
                            docs=docs    # Use the same extra_info for all paragraphs of the document
                        )                
                data.append(document)
    return data
def convert_nodes(data):
    name_space = 'criminal'
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, name_space=name_space)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        data, storage_context=storage_context
    )



pdf_directory = "data"  # Replace with your actual directory path
processed_data = process_pdfs(pdf_directory)  # Call the function once and store its result
convert_nodes(processed_data)  # Pass the result to the second function
