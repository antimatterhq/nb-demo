import json
from langchain.schema import Document
import langchain.schema.vectorstore
import base64
import antimatter
from typing import Optional

def create_or_load_domain():
    try:
        with open("domain_credentials.json", "r") as f:
            amr = antimatter.Session(**json.load(f))
            return amr
    except FileNotFoundError:
        amr = antimatter.new_domain("support@antimatter.io")
        with open("domain_credentials.json", "w") as f:
            json.dump({
                "domain": amr.domain_id,
                "api_key": amr.api_key,
            }, f)
        return amr

class AMRetriever(langchain.schema.retriever.BaseRetriever):

    delegate: Optional[langchain.schema.vectorstore.VectorStoreRetriever] = None
    amr: Optional["antimatter.Session"] = None
    read_context: Optional[str] = ""
    read_parameters: Optional[dict] = None

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def __init__(self, delegate: langchain.schema.vectorstore.VectorStoreRetriever,
                 antimatter: "antimatter.Session", read_context: str, read_parameters: dict):
        super().__init__()

        self.delegate = delegate
        self.amr = antimatter
        self.read_context = read_context
        self.read_parameters = read_parameters


    def _get_relevant_documents(self, query: str, *, run_manager):
        docs = self.delegate.get_relevant_documents(query, run_manager=run_manager)
        rv = []
        for d in docs:
            content = base64.b64decode(d.page_content)
            txt = (self.amr.load_capsule(data=content, read_context=self.read_context).
                   data(read_params=self.read_parameters))
            # print("returning raw data: ", txt)
            rv.append(Document(page_content=txt))
        return rv


def vector_retriever(amr:antimatter.Session, vr: langchain.schema.vectorstore.VectorStoreRetriever, read_context: str, read_parameters: dict):
    return AMRetriever(vr, amr, read_context, read_parameters)