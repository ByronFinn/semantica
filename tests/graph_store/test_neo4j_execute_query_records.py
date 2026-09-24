"""Regression tests for #1727: Neo4jStore.execute_query must convert
Node/Relationship record values via the Mapping protocol instead of
degrading them to a list of property names.

neo4j.graph.Node and neo4j.graph.Relationship subclass Entity, which
implements the Mapping protocol (items()/keys()) while __iter__ yields
property *keys*. The conversion in execute_query checked __iter__ before
items(), so ``MATCH (n) RETURN n`` came back as ``{"n": ["name", "age"]}``
and property values were silently lost. GraphStore.query and DecisionQuery
read decisions through this path.

The tests feed a real driver ``neo4j.graph.Node`` through execute_query
(mirroring the issue reproduction, no server needed), pin the unchanged
behavior for plain mappings, iterables, paths and primitives, and cover the
related ``GraphAnalytics.connected_components`` envelope read reported in
the same issue.

Requires the ``graph-neo4j`` extra (the same requirement as importing
Neo4jStore itself).
"""

import logging
import unittest
from types import MappingProxyType
from unittest.mock import MagicMock

from neo4j.graph import Graph, Node

from semantica.graph_store.graph_store import GraphAnalytics
from semantica.graph_store.neo4j_store import Neo4jStore
from semantica.utils.progress_tracker import get_progress_tracker


class _FakeRecord(dict):
    def keys(self):
        return list(super().keys())


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter([_FakeRecord(r) for r in self._rows])

    def keys(self):
        return list(self._rows[0].keys())

    def consume(self):
        summary = MagicMock()
        summary.counters = None
        summary.result_available_after = 0
        summary.result_consumed_after = 0
        return summary


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def run(self, query, parameters=None, **kwargs):
        return _FakeResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_store(rows):
    store = Neo4jStore.__new__(Neo4jStore)
    store.logger = logging.getLogger("test")
    store.progress_tracker = get_progress_tracker()
    store.get_session = lambda *a, **k: _FakeSession(rows)
    store.database = None
    return store


class _FakeNeo4jBackend:
    """Backend double whose class name routes GraphAnalytics to the Neo4j
    branch and whose execute_query returns the real envelope shape."""

    def __init__(self, envelope):
        self._envelope = envelope

    def execute_query(self, query, params):
        return self._envelope


class TestExecuteQueryRecordConversion(unittest.TestCase):
    def test_node_value_keeps_properties(self):
        node = Node(Graph(), "4:abc:1", 1, ["Person"], {"name": "Alice", "age": 30})
        store = _make_store([{"n": node, "name": "Alice"}])

        result = store.execute_query("MATCH (n:Person) RETURN n, n.name AS name")

        self.assertEqual(
            result["records"],
            [{"n": {"name": "Alice", "age": 30}, "name": "Alice"}],
        )

    def test_node_with_no_properties_converts_to_empty_dict(self):
        node = Node(Graph(), "4:abc:2", 2, ["Person"], {})
        store = _make_store([{"n": node}])

        result = store.execute_query("MATCH (n:Person) RETURN n")

        self.assertEqual(result["records"], [{"n": {}}])

    def test_mapping_value_keeps_entries(self):
        # MappingProxyType exposes items() and __iter__ exactly like the
        # driver's Node/Relationship entities (Relationship shares the
        # Entity/Mapping base with Node).
        mapping = MappingProxyType({"a": 1, "b": 2})
        store = _make_store([{"m": mapping}])

        result = store.execute_query("RETURN $m AS m", {"m": mapping})

        self.assertEqual(result["records"], [{"m": {"a": 1, "b": 2}}])

    def test_iterable_value_stays_a_list(self):
        store = _make_store([{"xs": [1, 2, 3]}])

        result = store.execute_query("RETURN [1, 2, 3] AS xs")

        self.assertEqual(result["records"], [{"xs": [1, 2, 3]}])

    def test_path_like_iterable_stays_a_list(self):
        class _PathLike:
            """Path objects are iterable and do not implement items()."""

            def __iter__(self):
                return iter(["n0", "r0", "n1"])

        store = _make_store([{"p": _PathLike()}])

        result = store.execute_query("MATCH p = (a)-[r]->(b) RETURN p")

        self.assertEqual(result["records"], [{"p": ["n0", "r0", "n1"]}])

    def test_string_and_primitive_values_pass_through(self):
        row = {"name": "Alice", "age": 30, "score": 1.5, "flag": True, "meta": None}
        store = _make_store([row])

        result = store.execute_query("RETURN name, age, score, flag, meta")

        self.assertEqual(result["records"], [dict(row)])


class TestConnectedComponentsEnvelope(unittest.TestCase):
    def test_reads_records_from_execute_query_envelope(self):
        envelope = {
            "success": True,
            "records": [
                {"componentId": 0, "nodes": [1, 2, 3]},
                {"componentId": 1, "nodes": [4]},
            ],
            "keys": ["componentId", "nodes"],
            "metadata": {"query": "CALL gds.wcc.stream(...)"},
        }

        components = GraphAnalytics(_FakeNeo4jBackend(envelope)).connected_components()

        self.assertEqual(
            components,
            [
                {"component": 0, "nodes": [1, 2, 3]},
                {"component": 1, "nodes": [4]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
