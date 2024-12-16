# RAG-Chat-Bot
Conversational RAG Chat Bot
# Streamlit Link - https://rag-chat-bot-jogendra-singh.streamlit.app/

This Streamlit app facilitates a **Conversational Retrieval-Augmented Generation (RAG)** system with Memory. It processes PDF documents, extracts their text, creates a retrieval chain, and allows users to query the information from the documents interactively. 
Below is a breakdown of the importance of each step in the code:

# 1. Library Imports and Configuration

Streamlit: For building the web-based UI.
langchain: This is a crucial library to handle document processing, embeddings, and building the retrieval system. It provides the logic to split, embed, and retrieve relevant content.
MistralAIEmbeddings & ChatMistralAI: These models are used for text embeddings and the chat model, respectively, which are essential for building the RAG chain that allows for context-based, intelligent query answering.
PyPDF2: Used to extract raw text from uploaded PDF files.
langgraph: This is used to manage the conversation's state, track history, and allow memory-based interaction.

# 2. Creating the RAG Chain (create_chain)

Text Processing:
Text Splitter: The text is split into chunks for efficient embedding and retrieval.
Mistral Embedding Model: Converts the text into vector embeddings, enabling the search for semantically similar content.
In-Memory Vector Store: Stores the embedded text, allowing fast retrieval of relevant chunks based on a query.
Retriever: Retrieves relevant document chunks based on the user's input, effectively finding context for the conversation.
RAG Chain Setup:
A chain is created where the model first reformulates the question to ensure that the context is clearly understood (using the history_aware_retriever), and then the answer is generated with the retrieved context.
Memory-Aware Execution: The system saves and updates the conversation’s state, allowing for context to be preserved and used in future interactions.

# 3. File Processing (file_processing)

PDF Upload and Extraction: Users upload PDFs, and the text is extracted for later processing. This step is crucial for building the knowledge base from which answers will be drawn.

# 4. Managing Chain Activation (enable_chain and disable_chain)

State Control: The system's chain is activated when PDFs are uploaded and processed, allowing the retrieval model to be built dynamically. The chain is disabled after query submission to ensure proper state management.

# 5. User Interaction (UI Components in Tabs)

Tab 1 (PDF Upload): Users can upload documents which trigger the creation of the RAG chain (if PDFs are uploaded and processed).
Tab 2 (QnA): Users can input queries, and the system uses the RAG chain to provide intelligent responses based on the uploaded PDFs.
Chat History: Interaction history is preserved, allowing for conversational flow. The history is displayed with streamlit_chat.message, keeping the UI intuitive and interactive.

# 6. Memory Management and State Handling

StateGraph: Tracks and manages the state of the conversation. This includes the user's input, chat history, context, and generated answers. It ensures that the RAG system can maintain the context over multiple exchanges with the user.
MemorySaver: Stores the interaction history and ensures that the context of the conversation is preserved for future interactions. This enables the system to be aware of previous exchanges.

# 7. Form Handling and Response Generation

Text Input Form: Users input questions regarding the uploaded PDFs. When submitted, the app processes the question and generates a response using the trained RAG system.
Response Generation: The call_model function invokes the RAG chain, processes the question with the history and context, and produces a concise answer.
Importance of Each Step:
PDF Extraction: This step allows the system to ingest external knowledge, transforming static documents into dynamic sources for answering user queries.
Text Chunking and Embedding: Ensures that the model can efficiently find relevant sections of the document, which is crucial for the retrieval-based approach.
Contextual Query Handling: Reformulating the query with context from previous interactions ensures that the system remains coherent across a multi-turn conversation.
Stateful Memory: The use of memory allows the model to "remember" previous interactions and context, creating a more natural, ongoing conversation rather than treating each query as independent.
Retrieval-Augmented Generation: This approach allows the system to combine the strengths of both retrieval (finding relevant information) and generation (crafting contextually appropriate responses), making it robust and scalable for various user interactions.

# Conclusion:

This system combines document processing, intelligent retrieval, and conversational AI with memory to create an interactive question-answering tool for uploaded PDFs. The use of stateful memory ensures that the system can handle dynamic, ongoing conversations with users, making it effective for knowledge-based tasks.
