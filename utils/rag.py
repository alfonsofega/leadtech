"""
RAG Pipeline using LlamaIndex
Processes PDFs and allows queries over them
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

# ===================== Config =====================
load_dotenv()

DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

client = OpenAI()


@dataclass
class SourcePreview:
    """
    Helper dataclass to represent the provenance of retrieved information.

    Attributes:
        file (str): Name of the source file (CV).
        score (float): Relevance score returned by the retriever.
        text (str): Preview snippet of the matched content.
    """
    file: str
    score: float
    text: str


class CVScreenerRAG:
    """
    Retrieval-Augmented Generation (RAG) pipeline for screening CVs stored as PDFs.

    This class handles:
        - Index creation or loading (vector store).
        - Query execution against the index.
        - Custom prompt configuration for flexible retrieval.
    """

    def __init__(self, cvs_dir="cvs",
                 persist_dir="data/vector_store",
                 llm_model: str = DEFAULT_LLM_MODEL,
                 embed_model: str = DEFAULT_EMBED_MODEL,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 5
                 ):
        """
        Initialize the RAG pipeline.

        Args:
            cvs_dir (str): Directory containing CV PDFs.
            persist_dir (str): Directory for persisting the vector index.
            llm_model (str): OpenAI LLM model to use for query answering.
            embed_model (str): OpenAI embedding model for vectorization.
            chunk_size (int): Size of text chunks for indexing.
            chunk_overlap (int): Overlap between consecutive text chunks.
            top_k (int): Number of top matches to return per query.
        """

        self.cvs_dir = cvs_dir
        self.persist_dir = persist_dir
        self.top_k = top_k

        # ---------- LLM + Embeddings (via LlamaIndex wrappers) ----------
        # Note: LlamaIndex recommends setting these globally via Settings
        Settings.llm = LlamaOpenAI(model=llm_model)
        Settings.embed_model = OpenAIEmbedding(model=embed_model)

        # ---------- Chunking ----------
        # SentenceSplitter provides configurable text splitting
        Settings.node_parser = SentenceSplitter.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.index = None
        self.query_engine = None

    def load_or_create_index(self):
        """
        Load an existing persisted index, or create a new one from PDF documents.

        - If index exists: loads from disk.
        - If not: reads all PDFs from `cvs_dir`, creates a vector index,
          and persists it in `persist_dir`.

        Also prepares a query engine for downstream use.
        """
        if os.path.exists(self.persist_dir):
            print("📂 Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("✅ Index loaded!")
        else:
            print("📄 Reading CVs from directory...")
            documents = SimpleDirectoryReader(
                self.cvs_dir,
                required_exts=[".pdf"],
            ).load_data()

            print(f"✅ Loaded {len(documents)} CV documents")
            print("🔨 Creating vector index (this may take a minute)...")

            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )

            print("💾 Saving index to disk...")
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("✅ Index created and saved!")

        # Initialize query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            response_mode="compact",  # concise answers
        )

    def _extract_sources(self, response) -> List[SourcePreview]:
        """
        Extract metadata and snippets from a query response.

        Args:
            response: Response object from the query engine.

        Returns:
            List[SourcePreview]: List of source file previews with relevance score and snippet.
        """
        sources: List[SourcePreview] = []
        for sn in getattr(response, "source_nodes", []) or []:
            meta = getattr(sn, "node", sn).metadata or {}
            # LlamaIndex usually exposes 'file_path'; fallback to 'file_name'
            file_name = meta.get("file_path") or meta.get("file_name") or "Unknown"
            text = (getattr(sn, "node", sn).text or "")[:200].strip()
            sources.append(
                SourcePreview(
                    file=os.path.basename(file_name),
                    score=round(sn.score or 0.0, 3),
                    text=(text + "...") if text else "",
                )
            )
        return sources

    def query(self, question):
        """
        Standard query against the RAG system.

        Args:
            question (str): User question.

        Returns:
            Tuple[str, List[dict]]: The textual answer and a list of source previews.
        """
        if self.query_engine is None:
            raise ValueError("Index not loaded! Call load_or_create_index() first")

        print(f"🔍 Searching for: {question}")
        response = self.query_engine.query(question)
        sources = self._extract_sources(response)
        return str(response), [s.__dict__ for s in sources]

    def custom_query(
            self,
            question: str,
            system_prompt: str | None = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Query the index with an optional custom system prompt.

        Args:
            question (str): User query.
            system_prompt (str, optional): Custom system prompt template.

        Returns:
            Tuple[str, List[dict]]: The textual answer and a list of source previews.
        """
        if self.query_engine is None:
            raise ValueError("Index not loaded! Call load_or_create_index() first")

        if system_prompt:
            from llama_index.core import PromptTemplate

            qa_prompt = PromptTemplate(
                (
                    f"{system_prompt}\n\n"
                    "Context information is below:\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge,\n"
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer:"
                )
            )

            qe = self.index.as_query_engine(
                text_qa_template=qa_prompt,
                similarity_top_k=self.top_k,
            )
            response = qe.query(question)
        else:
            response = self.query_engine.query(question)

        sources = self._extract_sources(response)
        return str(response), [s.__dict__ for s in sources]


def test_rag():
    """
    Basic test runner for the RAG pipeline.

    - Loads or creates the index.
    - Executes a set of test queries.
    - Prints answers and their sources.
    """
    print("🧪 Testing RAG Pipeline\n")

    rag = CVScreenerRAG()
    rag.load_or_create_index()

    test_queries = [
        "Who has Python experience?",
        "Which candidates worked at Google?",
        "Find someone with machine learning skills",
        "Who graduated from University of Jamaica?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}\nQ: {q}\n{'=' * 60}")
        answer, sources = rag.query(q)
        print(f"\nA: {answer}\n")
        print("Sources:")
        for src in sources:
            print(f"  📄 {src['file']} (relevance: {src['score']})")


if __name__ == "__main__":
    test_rag()
