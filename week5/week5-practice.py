# %% [markdown]
# # Assignment Items
# 
# 1. Demonstrate adding context to LLM query and then sending prompt to LLM -- done
# 2. Demonstrate: Loading Documents to langchain document loaders -> Chunking -> Converting to embeddings -> Storing in VectorDB -> Visualizing the embeddings
# 3. Langchain to bring it all together and perform RAG search
# 4. Using FAISS to Show size and Shape of VectorDB and perform similarity Search and again perform RAG search using Langchain
# 5. Proper Investigation i.e. Go Surgical

# %%
import gradio as gr
import os
import glob
import openai
from dotenv import load_dotenv
load_dotenv()

# %%
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key:
    print("API key available")
else:
    print("API key not available")

# %%
MODEL = "gpt-4o"

# %%
entire_context = {}

directory_structure = glob.glob("knowledge-base/**/*.md", recursive= True)
for document in directory_structure:
    # print(document.split("/")[-2])
    key = document.split("/")[-1].split(".md")[0]
    print(key)

    with open(document, "r") as file:
        entire_context[key] = file.read()

# %% [markdown]
# # With No Context at all:

# %%
query = "What are the offerings from Rellm?"
system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers.\n" \
" If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."


# %%
# messages = messages + [{"role": "user", "content": query}]
def chat(query, history):
    messages = [{"role":"system", "content": system_message}]
    messages = messages + [{"role": "user", "content": query}] + history
    stream = openai.chat.completions.create(model = MODEL,
                                     messages= messages,
                                     stream=True)
    print(history)
    response = ''
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response

view = gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# # With Entire Knowledge Base as Context

# %%
def add_context(messages, context):
    messages = messages + "\n\nThe following context maybe relevant in answering:\n\n " + context
    return(messages)

def provide_relevant_context(messages, entire_context):
    relevant_context = ""

    for doc, knowledge in entire_context.items():
        if doc in messages:
            relevant_context += knowledge
    
    return(relevant_context)

# %%
# messages = messages + [{"role": "user", "content": query}]

def chat(query, history):
    messages = [{"role":"system", "content": system_message}]
    query = add_context(query, str(entire_context))
    messages = messages + [{"role": "user", "content": query}] + history
    stream = openai.chat.completions.create(model = MODEL,
                                     messages= messages,
                                     stream=True)
    print(history)
    response = ''
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response

view = gr.ChatInterface(chat, type="messages").launch()
    

# %% [markdown]
# # With Relevant Context as Context

# %%

def chat(query, history):
    messages = [{"role":"system", "content": system_message}]
    relevant_context = provide_relevant_context(query, entire_context)
    print(f"Relevant Context: {relevant_context}")
    query = add_context(query, relevant_context)
    messages = messages + [{"role": "user", "content": query}] + history
    stream = openai.chat.completions.create(model = MODEL,
                                     messages= messages,
                                     stream=True)
    # print(history)
    response = ''
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response

view = gr.ChatInterface(chat, type="messages").launch()

# %% [markdown]
# # Now with langchain and FAISS similarity searches

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
model.invoke("Hi")

# %%
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter


loader = DirectoryLoader("../", glob="**/*.md", show_progress=True , silent_errors=True, use_multithreading=True, loader_cls=TextLoader)
documents = loader.load()

print(documents[50].page_content[:100])
len(documents)

# %%
textsplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_documents = textsplitter.split_documents(documents)

# %%
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()

# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch

if os.path.exists("vector_db"):
    Chroma(persist_directory="vector_db", embedding_function=embeddings).delete_collection()

vector_store = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(), persist_directory="vector_db")

# %%
collections = vector_store._collection
sample_embedding = collections.get(limit=1, include=["embeddings"])["embeddings"][0]
len(sample_embedding)

# %%
# Hence 66x1536

# %% [markdown]
# # Visualization of Vector Embeddings [Skipping for now]

# %%
vector_store.similarity_search("What are the offerings of Rellm?")

# %% [markdown]
# # Bringing it together with Langchain

# %%
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model = MODEL, temperature= 0.9)
retriever = vector_store.as_retriever(search_kwargs={'k':10})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

conversation_chain = ConversationalRetrievalChain.from_llm(llm = model,  retriever = retriever, memory = memory)

def chat(message, history):
    result = conversation_chain.invoke({'question': message})
    return(result['answer'])

gr.ChatInterface(chat, type = "messages").launch(inbrowser = False)

# %%


# %%



