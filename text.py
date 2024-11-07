from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from colorama import Fore
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Initialize embeddings and model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""

prompt_template = ChatPromptTemplate.from_template(template)

def load_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    try:
        raw_documents = TextLoader("./docs/user-manual.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        return text_splitter.split_documents(raw_documents)
    except Exception as e:
        print(Fore.RED + "Error loading documents: " + str(e))
        traceback.print_exc()
        return []

def load_embeddings(documents):
    """Create a vector store from a set of documents."""
    try:
        db = Chroma.from_documents(documents, embeddings)
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        )
    except Exception as e:
        print(Fore.RED + "Error loading embeddings: " + str(e))
        traceback.print_exc()
        return None

def generate_response(retriever, query):
    """Generate a response using the given retriever and query."""
    try:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
        )
        return chain.invoke(query)
    except Exception as e:
        print(Fore.RED + "Error generating response: " + str(e))
        traceback.print_exc()
        return "Sorry, an error occurred while generating the response."

def query(query):
    """Handle the query, loading documents, embeddings, and generating a response."""
    documents = load_documents()
    if not documents:
        return "No documents found to load."
    
    retriever = load_embeddings(documents)
    if retriever is None:
        return "Failed to create retriever from documents."
    
    response = generate_response(retriever, query)
    return response

def main():
    """Main loop for handling user input indefinitely."""
    print(Fore.WHITE + "---------------------------------------------------------------------------")
    while True:
        user_label = Fore.BLUE + "\n\x1B[3mUser /> " + "\x1B[0m" + Fore.RESET
        user_question = input(user_label)
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        response = query(user_question)
        print(Fore.GREEN + "Bot /> " + response)
        print(Fore.WHITE + "---------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
