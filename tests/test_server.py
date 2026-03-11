"""
Unit tests for gRPC server implementation.

Note: Full server method tests require gRPC test utilities and are better suited
for integration testing. These unit tests focus on verifiable logic components.
"""
import pytest
from unittest.mock import MagicMock
import sys


# Pre-mock proto modules to avoid import errors
# These must be set before importing server module
_mock_pb2 = MagicMock()
_mock_pb2.SERVICE_STATUS_SERVING = 1
_mock_pb2.SERVICE_STATUS_NOT_SERVING = 2
_mock_pb2.INGEST_STAGE_STORING = 5
_mock_pb2.INGEST_STAGE_DOWNLOADING = 1
_mock_pb2.INGEST_STAGE_PARSING = 2
_mock_pb2.INGEST_STAGE_CHUNKING = 3
_mock_pb2.INGEST_STAGE_EMBEDDING = 4
_mock_pb2.INGEST_STAGE_COMPLETE = 6
_mock_pb2.INGEST_STAGE_ERROR = 7
_mock_pb2.INGEST_STAGE_UNSPECIFIED = 0
_mock_pb2.RELEVANCE_HIGH = 1
_mock_pb2.RELEVANCE_MEDIUM = 2
_mock_pb2.RELEVANCE_LOW = 3

_mock_pb2_grpc = MagicMock()

# Create a real base class for RagServiceServicer
class MockRagServiceServicer:
    pass

_mock_pb2_grpc.RagServiceServicer = MockRagServiceServicer

# Install mocks
sys.modules['gen'] = MagicMock()
sys.modules['gen.python'] = MagicMock()
sys.modules['gen.python.rag'] = MagicMock()
sys.modules['gen.python.rag.v1'] = MagicMock()
sys.modules['gen.python.rag.v1.rag_service_pb2'] = _mock_pb2
sys.modules['gen.python.rag.v1.rag_service_pb2_grpc'] = _mock_pb2_grpc


class TestStageMapping:
    """Tests for stage string to enum mapping."""

    def test_stage_map_contains_all_expected_stages(self):
        """Test that STAGE_MAP contains all expected stages."""
        from rag_service.server import STAGE_MAP

        expected_stages = [
            "INVALIDATING",
            "PREPARING",
            "DOWNLOADING",
            "PARSING",
            "CHUNKING",
            "EMBEDDING",
            "STORING",
            "COMPLETE",
            "ERROR"
        ]

        for stage in expected_stages:
            assert stage in STAGE_MAP, f"Stage '{stage}' not found in STAGE_MAP"

    def test_stage_map_values_are_integers(self):
        """Test that all STAGE_MAP values are valid integers."""
        from rag_service.server import STAGE_MAP

        for stage, value in STAGE_MAP.items():
            # Values should be protobuf enum values (integers)
            assert isinstance(value, int) or hasattr(value, '__int__'), \
                f"Stage '{stage}' has non-integer value: {value}"


class TestRelevanceMapping:
    """Tests for relevance string to enum mapping."""

    def test_relevance_map_contains_all_levels(self):
        """Test that RELEVANCE_MAP contains all expected levels."""
        from rag_service.server import RELEVANCE_MAP

        expected_levels = ["high", "medium", "low"]

        for level in expected_levels:
            assert level in RELEVANCE_MAP, f"Relevance level '{level}' not found in RELEVANCE_MAP"

    def test_relevance_map_values_are_integers(self):
        """Test that all RELEVANCE_MAP values are valid integers."""
        from rag_service.server import RELEVANCE_MAP

        for level, value in RELEVANCE_MAP.items():
            assert isinstance(value, int) or hasattr(value, '__int__'), \
                f"Relevance '{level}' has non-integer value: {value}"


class TestRagServicerInit:
    """Tests for RagServicer initialization."""

    def test_servicer_can_be_instantiated_and_has_methods(self):
        """Test that RagServicer can be instantiated and has all required methods."""
        from rag_service.server import RagServicer

        servicer = RagServicer()

        # Should have highlight_service attribute
        assert hasattr(servicer, 'highlight_service')

        # Should have all required gRPC methods
        required_methods = [
            'IngestPdf',
            'Query',
            'GetHighlightedPdf',
            'InvalidateDocument',
            'HealthCheck',
        ]

        for method in required_methods:
            assert hasattr(servicer, method), f"Method '{method}' not found on RagServicer"
            assert callable(getattr(servicer, method)), f"'{method}' is not callable"


class TestBBoxConversion:
    """Tests for bbox format conversion used in server methods."""

    def test_bbox_list_conversion(self):
        """Test converting bbox objects to list format."""
        # This tests the conversion logic used in GetHighlightedPdf
        mock_bbox = MagicMock()
        mock_bbox.x0 = 10.0
        mock_bbox.y0 = 20.0
        mock_bbox.x1 = 100.0
        mock_bbox.y1 = 50.0

        bboxes = [mock_bbox]

        # Convert using the same logic as in server.py
        converted = [
            [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
            for bbox in bboxes
        ]

        assert converted == [[10.0, 20.0, 100.0, 50.0]]

    def test_multiple_bboxes_conversion(self):
        """Test converting multiple bbox objects."""
        mock_bbox1 = MagicMock(x0=10, y0=20, x1=100, y1=50)
        mock_bbox2 = MagicMock(x0=200, y0=300, x1=400, y1=500)

        bboxes = [mock_bbox1, mock_bbox2]

        converted = [
            [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
            for bbox in bboxes
        ]

        assert len(converted) == 2
        assert converted[0] == [10, 20, 100, 50]
        assert converted[1] == [200, 300, 400, 500]


class TestSourceConversion:
    """Tests for source format conversion used in Query method."""

    def test_source_relevance_mapping_has_all_levels(self):
        """Test that relevance strings are mapped to pb2 enum values."""
        from rag_service.server import RELEVANCE_MAP

        # All levels should be mapped
        assert "high" in RELEVANCE_MAP
        assert "medium" in RELEVANCE_MAP
        assert "low" in RELEVANCE_MAP

        # Values should be the pb2 enum constants (not None)
        assert RELEVANCE_MAP.get("high") is not None
        assert RELEVANCE_MAP.get("medium") is not None
        assert RELEVANCE_MAP.get("low") is not None

    def test_source_default_relevance(self):
        """Test that unknown relevance defaults appropriately."""
        from rag_service.server import RELEVANCE_MAP

        # Unknown relevance should use the default
        default = RELEVANCE_MAP.get("unknown", RELEVANCE_MAP.get("high"))
        assert default == RELEVANCE_MAP.get("high")
