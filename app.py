import streamlit as st
from typing import Iterator
import os
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

try:
    from llama_index import Document
    from llama_index.text_splitter import SentenceSplitter
except ImportError:
    from llama_index.core import Document
    from llama_index.core.text_splitter import SentenceSplitter

openai.api_key = ''

# Document paths
document_infos = {
    "Cardiovascular / respiratory": "Knowledge_Base/Cardiovascular_respiratory.pdf",
    "Eyes": "Knowledge_Base/Eyes.pdf",
    "Foot / toes": "Knowledge_Base/Foot_toes.pdf",
    "Head": "Knowledge_Base/Head.pdf",
    "Hand / fingers": "Knowledge_Base/Hand_fingers.pdf",
    "Low back": "Knowledge_Base/Low_back.pdf",
    "Lower arm": "Knowledge_Base/Lower_arm.pdf",
    "Lower leg": "Knowledge_Base/Lower_leg.pdf"
}

# Load documents
documents = []
for name, path in document_infos.items():
    docs = SimpleDirectoryReader(input_files=[path]).load_data()
    for doc in docs:
        doc.metadata['name'] = name
        documents.append(doc)

# Initialize LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Create service context
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

# Filter function
def filter_documents_by_label(documents, label):
    return [doc for doc in documents if label.lower() in doc.metadata['name'].lower()]

# Predictor class
class IncidentCodePredictor:
    def __init__(self, documents, service_context):
        self.documents = documents
        self.service_context = service_context

    def predict_code(self, label, query):
        filtered_documents = filter_documents_by_label(self.documents, label)
        if not filtered_documents:
            return f"No documents found for label: {label}"
        index = VectorStoreIndex.from_documents(filtered_documents, service_context=self.service_context)
        query_engine = index.as_query_engine()
        prompt = f"""What labels does the document suggest for this incident description? Return only the two most likely labels (if the document has more than two labels) and your confidence score percentages for each. Do not use preambles. Write only the label names as they appear in the document. Format the output as follows: A, n% - B, n%. Ensure your answer is grounded in the document guidelines. Remember: DO NOT make up an answer. CRUCIAL: The confidence scores must FACTUALLY reflect how confident you are in the labels  correctness. {query}"""
        response = query_engine.query(prompt)
        return response.text if hasattr(response, "text") else str(response)

# Streamlit app
st.title("Incident Code Predictor")
st.write("Select a label and input a query to get the predicted incident codes.")

label = st.selectbox("Select Label", list(document_infos.keys()))
query_text = st.text_area("Enter Query Text", "")

if st.button("Predict Code"):
    predictor = IncidentCodePredictor(documents, service_context)
    response = predictor.predict_code(label, query_text)
    st.write("Prediction Response:")
    st.write(response)