import streamlit as st
from streamlit_chat import message as msg
import os
from dotenv import load_dotenv

import numpy as np
# to read from PDFs
from PyPDF2 import PdfReader
# for chunking the texts from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
# for embedding
from langchain_mistralai.embeddings import MistralAIEmbeddings
# for vector store
from langchain_core.vectorstores import InMemoryVectorStore
# chat model to use in the RAG
from langchain_mistralai.chat_models import ChatMistralAI
# to use custom prompts and Messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# to build document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# to create retriever chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# to add memory to RAG
from typing import Sequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict



def create_chain(text):
    text_splitter = RecursiveCharacterTextSplitter()
    text = text_splitter.split_text(text=text)
        
    # Initialize the embeddings model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_KEY)
        
    # create vectorstore using ChromaDB
    vectorstore = InMemoryVectorStore.from_texts(
            texts=text, embedding=embeddings
        )
    retriever = vectorstore.as_retriever()

    #Initialize llm
    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=API_KEY)
        
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say politely that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    
    st.session_state['rag_chain'] = rag_chain
    
    # ### Statefully manage chat history ###

    # # Updates to input, context and answer strings will overwrite the strings
    # # Update to the message (input and answer) will get chat_history updated by 
    # # appending the messages to chat_history
    # class State(TypedDict):
    #     input: str
    #     chat_history: Annotated[Sequence[BaseMessage], add_messages]
    #     context: str
    #     answer: str

    # # Define the function that calls the model
    # def call_model(state: State):
    #     response = rag_chain.invoke(state)
    #     return {
    #         "chat_history": [
    #             HumanMessage(state["input"]),
    #             AIMessage(response["answer"]),
    #         ],
    #         "context": response["context"],
    #         "answer": response["answer"],
    #     }
    
    # Define a new graph
    workflow = StateGraph(state_schema=State)

    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return(app)

### Statefully manage chat history ###

# Updates to input, context and answer strings will overwrite the strings
# Update to the message (input and answer) will get chat_history updated by 
# appending the messages to chat_history
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# Define the function that calls the model
def call_model(state: State):
    rag_chain = st.session_state['rag_chain']
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


def file_processing(pdfs):
    text=""
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return(text)

def enable_chain():
    st.session_state['process_chain'] = 1    

def disable_chain():
    st.session_state['process_chain'] = 0 

if __name__ == "__main__":
    #load_dotenv()
    #API_KEY = os.getenv("MISTRAL_API_KEY")
    
    API_KEY = st.secrets["MISTRAL_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]
    
    os.environ["MISTRAL_API_KEY"] = API_KEY
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Very first time, setting the config
    if 'process_chain'not in st.session_state:
        st.session_state['process_chain'] = 0
        
    
    st.title("Conversational RAG with Memory")
    
    tab1, tab2 = st.tabs(["PDF Upload","QnA"])
    
    tab1.subheader("Document Processing")
    with tab1:
        pdfs = tab1.file_uploader("Upload files", accept_multiple_files=True, on_change=enable_chain)
        text=""
        text = file_processing(pdfs)
        
        if pdfs:
            # When question is available and new documents uploaded
            if (len(text) !=0) and (st.session_state['process_chain'] !=0):      
                app = create_chain(text)
                st.session_state['chain'] = app
                #st.session_state['vectorstore'] = vectorstore
        else:
            if 'chain' in st.session_state:
                del st.session_state['chain']
                #del st.session_state['vectorstore']    
    
    tab2.subheader("Chat with your PDFs here")
    with tab2:
        with st.form(key='user_form', clear_on_submit=True):
            user_input = st.text_input(label="Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=disable_chain)
            
            if submit_button and user_input:
                if 'chain' in st.session_state:
                    with st.spinner('Generating response...'):
                        app = st.session_state['chain']
                        # setting config is important to handle multiple users
                        # Also the value is referred by the Memory CheckPointer
                        config = {"configurable": {"thread_id": "thread1"}}
                        result = app.invoke(
                            {"input": user_input},
                            config=config,
                        )
                        #st.write(result["answer"])  
                        # writing the chat history using streamlit chat message format
                        #with st.container:
                        chat_history = app.get_state(config).values["chat_history"]
                        for message in chat_history:
                            if message.type == 'human':
                                msg(message.content, is_user=True)
                            else:
                                msg(message.content)
                
                else:
                    st.write("RAG chain not built, upload the PDFS and comeback.")