from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from  langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


Path2= 'PlacementChronicles.pdf'
Path1 = 'SIChronicles.pdf'

genai.configure(api_key=GOOGLE_API_KEY)


# Loading the data

def load_documents(Path):
	loader = PyPDFLoader(Path)
	documents = loader.load()
	return documents

# Splitting the Long data into Different chunks

text_splitter = RecursiveCharacterTextSplitter(
									  chunk_size=1000,
									  chunk_overlap=500,
									  length_function=len,
									  add_start_index=True,
)

documents = load_documents(Path2) + load_documents(Path1)

chunks = text_splitter.split_documents(documents)
print(f"\n\n Just split {len(documents)} into {len(chunks)} chunks\n\n")

CHROMA_PATH = "chroma"

# Creating a new DB from the documents

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)
retriever = Chroma.from_documents(
					  chunks,
					  GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY),
					  persist_directory=CHROMA_PATH
).as_retriever(search_kwargs={"k":1})

print(f"Saved {len(chunks)} to {CHROMA_PATH}.")



print("\n\n")

# Creating a Chain To invoke Answers to Queries

template = """
You are an expert at finding answers from given context. Use the Question Provided to find answers in the below Context


<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chain = create_stuff_documents_chain(llm, prompt)


while True:
	## Getting Relevant docs from Vector Database using Query
	print("\n\nEnter your query: ")
	Q = input()
	print("\n\n")

	if Q == "q":
		break

	docs = retriever.invoke(Q)

	result = chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content=Q)
        ],
    }
)
	print("Here's the result : ")
	print(result)


