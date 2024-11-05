from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from colorama import Fore
from dotenv import load_dotenv


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:"""



prompt_template = ChatPromptTemplate.from_template(template)



def load_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_documents = TextLoader("./docs/user-manual.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    return text_splitter.split_documents(raw_documents)



def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    db = Chroma.from_documents(documents, embeddings)
    return db.as_retriever( 
        search_type="similarity",
        search_kwargs={"k": 1})



def generate_response(retriever, query):
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    # create the prompt

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain.invoke(query)



def query(query):
    documents = load_documents()
    retriever = load_embeddings(documents, query)
    response = generate_response(retriever, query)
    return response

def start():
    instructions = (
        "User /> "
    )
    user_label = (Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)
    user_question = input(user_label)
    response = query(user_question)

    print(Fore.GREEN + f"Bot /> " + response )
    print(Fore.WHITE + "---------------------------------------------------------------------------")
    start()

def main():
    # Main loop for handling user input indefinitely
    while True:
        user_label = (Fore.BLUE + "\n\x1B[3m" + "User /> " + "\x1B[0m" + Fore.RESET)
        user_question = input(user_label)
        if user_question.lower() == 'exit':
            print("Exiting...")
            break
        response = query(user_question)
        print(Fore.GREEN + f"Bot /> " + response )
        print(Fore.WHITE + "---------------------------------------------------------------------------")


if __name__ == "__main__":
    print(Fore.WHITE + "---------------------------------------------------------------------------")
    main()
