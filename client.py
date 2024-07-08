from retriever import load_embedding_model, load_reranker_model
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, PromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.groq import Groq
import os


class RAGClient:
    llm = None
    embedding_model = None
    reranker_model = None
    docs = None
    index = None
    query_engine = None

    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-large",
    ):
        self.llm = Groq(model=model_name, api_key=os.environ.get("GROQ_API_KEY"))
        self.embedding_model = LangchainEmbedding(load_embedding_model(model_name=embedding_model_name))
        self.reranker_model = load_reranker_model(reranker_model_name=reranker_model_name)

    def read_files(self, input_dir_path: str):
        self.loader = SimpleDirectoryReader(
            input_dir=input_dir_path,
            required_exts=[".py", ".ipynb", ".js", ".ts", ".md", ".pdf"],
            recursive=True,
        )
        self.docs = self.loader.load_data()
        if not self.docs:
            raise Exception("No documents found in the provided directory.")
        return True

    def generate_index(self):
        Settings.embed_model = self.embedding_model
        self.index = VectorStoreIndex.from_documents(self.docs)

    def create_query_engine(self):
        Settings.llm = self.llm
        self.query_engine = self.index.as_query_engine(streaming=True, similarity_top_k=4)

        qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are llama3, a large language model developed by Meta AI. Revanth has integrated you into this environment to answer the user's coding questions. Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
        )

        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
    
