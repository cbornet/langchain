import pytest

from langchain_core.documents import Document
from langchain_core.graph_vectorstores.base import _texts_to_documents
from langchain_core.graph_vectorstores.links import Link


def test_texts_to_documents() -> None:
    link1 = [Link.incoming(kind="hyperlink", tag="http://b")]
    link2 = [Link.outgoing(kind="url", tag="http://c")]
    assert list(
        _texts_to_documents(
            ["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"], [link1, link2]
        )
    ) == [
        Document(id="a", metadata={"a": "b", "links": link1}, page_content="a"),
        Document(id="b", metadata={"c": "d", "links": link2}, page_content="b"),
    ]
    assert list(_texts_to_documents(["a", "b"], None, ["a", "b"], [link1, link2])) == [
        Document(id="a", metadata={"links": link1}, page_content="a"),
        Document(id="b", metadata={"links": link2}, page_content="b"),
    ]
    assert list(
        _texts_to_documents(["a", "b"], [{"a": "b"}, {"c": "d"}], None, [link1, link2])
    ) == [
        Document(metadata={"a": "b", "links": link1}, page_content="a"),
        Document(metadata={"c": "d", "links": link2}, page_content="b"),
    ]
    assert list(
        _texts_to_documents(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"], None)
    ) == [
        Document(id="a", metadata={"a": "b", "links": []}, page_content="a"),
        Document(id="b", metadata={"c": "d", "links": []}, page_content="b"),
    ]
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a", "b"], None, ["a"], None))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a", "b"], [{"a": "b"}], None, None))
        with pytest.raises(ValueError):
            list(_texts_to_documents(["a", "b"], None, None, [link1]))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a"], [{"a": "b"}, {"c": "d"}], None, None))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a"], None, ["a", "b"], None))
    with pytest.raises(ValueError):
        list(_texts_to_documents(["a"], None, None, [link1, link2]))
