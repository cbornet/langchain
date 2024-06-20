import math
import os
from typing import Iterable, List, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_stores import CassandraGraphStore

CASSANDRA_DEFAULT_KEYSPACE = "graph_test_keyspace"


def _get_graph_store(
    embedding_class: Type[Embeddings], documents: Iterable[Document] = ()
) -> CassandraGraphStore:
    import cassio
    from cassandra.cluster import Cluster
    from cassio.config import check_resolve_session, resolve_keyspace

    node_table = "graph_test_node_table"
    edge_table = "graph_test_edge_table"

    if any(
        env_var in os.environ
        for env_var in [
            "CASSANDRA_CONTACT_POINTS",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_INIT_STRING",
        ]
    ):
        cassio.init(auto=True)
        session = check_resolve_session()
    else:
        cluster = Cluster()
        session = cluster.connect()
    keyspace = resolve_keyspace() or CASSANDRA_DEFAULT_KEYSPACE
    cassio.init(session=session, keyspace=keyspace)
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    session.execute(f"DROP TABLE IF EXISTS {keyspace}.{node_table}")
    session.execute(f"DROP TABLE IF EXISTS {keyspace}.{edge_table}")
    store = CassandraGraphStore.from_documents(
        documents,
        embedding=embedding_class(),
        session=session,
        keyspace=keyspace,
        node_table=node_table,
        edge_table=edge_table,
    )
    return store


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class AngularTwoDimensionalEmbeddings(Embeddings):
    """
    From angles (as strings in units of pi) to unit embedding vectors on a circle.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !
        """
        try:
            angle = float(text)
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
            return [0.0, 0.0]


def _result_ids(docs: Iterable[Document]) -> List[str]:
    from ragstack_knowledge_store.graph_store import CONTENT_ID

    return list(map(lambda d: d.metadata[CONTENT_ID], docs))


def test_link_directed() -> None:
    from ragstack_knowledge_store.link_tag import IncomingLinkTag, OutgoingLinkTag

    a = Document(
        page_content="A",
        metadata={
            "content_id": "a",
            "link_tags": {
                IncomingLinkTag(kind="hyperlink", tag="http://a"),
            },
        },
    )
    b = Document(
        page_content="B",
        metadata={
            "content_id": "b",
            "link_tags": {
                IncomingLinkTag(kind="hyperlink", tag="http://b"),
                OutgoingLinkTag(kind="hyperlink", tag="http://a"),
            },
        },
    )
    c = Document(
        page_content="C",
        metadata={
            "content_id": "c",
            "link_tags": {
                OutgoingLinkTag(kind="hyperlink", tag="http://a"),
            },
        },
    )
    d = Document(
        page_content="D",
        metadata={
            "content_id": "d",
            "link_tags": {
                OutgoingLinkTag(kind="hyperlink", tag="http://a"),
                OutgoingLinkTag(kind="hyperlink", tag="http://b"),
            },
        },
    )

    store = _get_graph_store(FakeEmbeddings, [a, b, c, d])

    assert list(store.store._linked_ids("a")) == []
    assert list(store.store._linked_ids("b")) == ["a"]
    assert list(store.store._linked_ids("c")) == ["a"]
    assert sorted(store.store._linked_ids("d")) == ["a", "b"]


def test_mmr_traversal() -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    from ragstack_knowledge_store.link_tag import IncomingLinkTag, OutgoingLinkTag

    store = _get_graph_store(AngularTwoDimensionalEmbeddings)

    v0 = Document(
        page_content="-0.124",
        metadata={
            "content_id": "v0",
            "link_tags": {
                OutgoingLinkTag(kind="explicit", tag="link"),
            },
        },
    )
    v1 = Document(
        page_content="+0.127",
        metadata={
            "content_id": "v1",
        },
    )
    v2 = Document(
        page_content="+0.25",
        metadata={
            "content_id": "v2",
            "link_tags": {
                IncomingLinkTag(kind="explicit", tag="link"),
            },
        },
    )
    v3 = Document(
        page_content="+1.0",
        metadata={
            "content_id": "v3",
            "link_tags": {
                IncomingLinkTag(kind="explicit", tag="link"),
            },
        },
    )
    store.add_documents([v0, v1, v2, v3])

    results = store.mmr_traversal_search("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = store.mmr_traversal_search("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = store.mmr_traversal_search("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = store.mmr_traversal_search("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = store.mmr_traversal_search("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


def test_write_retrieve_keywords() -> None:
    from langchain_openai import OpenAIEmbeddings
    from ragstack_knowledge_store.link_tag import (
        BidirLinkTag,
        IncomingLinkTag,
        OutgoingLinkTag,
    )

    greetings = Document(
        page_content="Typical Greetings",
        metadata={
            "content_id": "greetings",
            "link_tags": {
                IncomingLinkTag(kind="parent", tag="parent"),
            },
        },
    )
    doc1 = Document(
        page_content="Hello World",
        metadata={
            "content_id": "doc1",
            "link_tags": {
                OutgoingLinkTag(kind="parent", tag="parent"),
                BidirLinkTag(kind="kw", tag="greeting"),
                BidirLinkTag(kind="kw", tag="world"),
            },
        },
    )
    doc2 = Document(
        page_content="Hello Earth",
        metadata={
            "content_id": "doc2",
            "link_tags": {
                OutgoingLinkTag(kind="parent", tag="parent"),
                BidirLinkTag(kind="kw", tag="greeting"),
                BidirLinkTag(kind="kw", tag="earth"),
            },
        },
    )
    store = _get_graph_store(OpenAIEmbeddings, [greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also
    # shows up.
    results: Iterable[Document] = store.similarity_search("Earth", k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = store.similarity_search("Earth", k=1)
    assert _result_ids(results) == ["doc2"]

    results = store.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = store.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = store.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via
    # keyword edge.
    results = store.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}
