import json
import unittest
from unittest.mock import MagicMock, patch

from genai_otel.config import OTelConfig
from genai_otel.instrumentors.bedrock_agents_instrumentor import BedrockAgentsInstrumentor


class TestBedrockAgentsInstrumentor(unittest.TestCase):
    """Tests for BedrockAgentsInstrumentor"""

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_init_with_bedrock_agents_available(self, mock_logger):
        """Test that __init__ detects Bedrock Agents availability."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = [
            "bedrock-agent-runtime",
            "bedrock",
            "s3",
        ]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            self.assertTrue(instrumentor._bedrock_agents_available)
            mock_logger.debug.assert_called_with(
                "AWS Bedrock Agents runtime detected and available for instrumentation"
            )

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_init_with_bedrock_agents_not_available(self, mock_logger):
        """Test that __init__ handles missing Bedrock Agents service."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["s3", "ec2"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            self.assertFalse(instrumentor._bedrock_agents_available)
            mock_logger.debug.assert_called_with("AWS Bedrock Agents runtime service not available")

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_init_with_boto3_not_installed(self, mock_logger):
        """Test that __init__ handles missing boto3."""
        with patch.dict("sys.modules", {"boto3": None}):
            instrumentor = BedrockAgentsInstrumentor()

            self.assertFalse(instrumentor._bedrock_agents_available)
            mock_logger.debug.assert_called_with(
                "boto3 not installed, Bedrock Agents instrumentation will be skipped"
            )

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_instrument_when_bedrock_agents_not_available(self, mock_logger):
        """Test that instrument skips when Bedrock Agents is not available."""
        with patch.dict("sys.modules", {"boto3": None}):
            instrumentor = BedrockAgentsInstrumentor()
            config = MagicMock()

            instrumentor.instrument(config)

            mock_logger.debug.assert_any_call(
                "Skipping Bedrock Agents instrumentation - library not available"
            )

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_instrument_wraps_base_client(self, mock_logger):
        """Test that instrument wraps BaseClient._make_request method."""
        # Create mock boto3 module
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        # Create mock botocore
        mock_botocore = MagicMock()
        mock_base_client = type("BaseClient", (), {})
        mock_base_client._make_request = MagicMock()
        mock_botocore.client.BaseClient = mock_base_client

        mock_wrapt = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.client": mock_botocore.client,
                "wrapt": mock_wrapt,
            },
        ):
            instrumentor = BedrockAgentsInstrumentor()
            config = MagicMock()

            # Call instrument
            instrumentor.instrument(config)

            # Assert
            self.assertEqual(instrumentor.config, config)
            self.assertTrue(instrumentor._instrumented)
            mock_logger.info.assert_called_with("AWS Bedrock Agents instrumentation enabled")

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_false(self, mock_logger):
        """Test that exceptions are logged when fail_on_error is False."""
        # Create mock that raises
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        mock_botocore = MagicMock()
        mock_botocore_client = MagicMock()
        type(mock_botocore_client).BaseClient = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict(
            "sys.modules",
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.client": mock_botocore_client,
                "wrapt": MagicMock(),
            },
        ):
            instrumentor = BedrockAgentsInstrumentor()
            config = MagicMock()
            config.fail_on_error = False

            # Should not raise
            instrumentor.instrument(config)

            mock_logger.error.assert_called_once()

    @patch("genai_otel.instrumentors.bedrock_agents_instrumentor.logger")
    def test_instrument_exception_with_fail_on_error_true(self, mock_logger):
        """Test that exceptions are raised when fail_on_error is True."""
        # Create mock that raises
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        mock_botocore = MagicMock()
        mock_botocore_client = MagicMock()
        type(mock_botocore_client).BaseClient = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access failed"))
        )

        with patch.dict(
            "sys.modules",
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.client": mock_botocore_client,
                "wrapt": MagicMock(),
            },
        ):
            instrumentor = BedrockAgentsInstrumentor()
            config = MagicMock()
            config.fail_on_error = True

            # Should raise
            with self.assertRaises(RuntimeError) as context:
                instrumentor.instrument(config)

            self.assertEqual(str(context.exception), "Access failed")

    def test_extract_invoke_agent_attributes(self):
        """Test extraction of invoke_agent attributes."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create request dict
            request_dict = {
                "body": json.dumps(
                    {
                        "agentId": "test-agent-123",
                        "agentAliasId": "alias-456",
                        "sessionId": "session-789",
                        "inputText": "What is the weather today?",
                        "enableTrace": True,
                        "sessionState": {
                            "promptSessionAttributes": {"context": "weather"},
                            "sessionAttributes": {"location": "SF"},
                        },
                    }
                )
            }

            attrs = instrumentor._extract_invoke_agent_attributes(request_dict)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "bedrock_agents")
            self.assertEqual(attrs["gen_ai.operation.name"], "invoke_agent")
            self.assertEqual(attrs["bedrock.agent.id"], "test-agent-123")
            self.assertEqual(attrs["bedrock.agent.alias_id"], "alias-456")
            self.assertEqual(attrs["bedrock.agent.session_id"], "session-789")
            self.assertEqual(attrs["bedrock.agent.input_text"], "What is the weather today?")
            self.assertTrue(attrs["bedrock.agent.enable_trace"])

    def test_extract_retrieve_attributes(self):
        """Test extraction of retrieve attributes."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create request dict
            request_dict = {
                "body": json.dumps(
                    {
                        "knowledgeBaseId": "kb-12345",
                        "retrievalQuery": {"text": "company benefits policy"},
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": 5,
                                "overrideSearchType": "HYBRID",
                            }
                        },
                    }
                )
            }

            attrs = instrumentor._extract_retrieve_attributes(request_dict)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "bedrock_agents")
            self.assertEqual(attrs["gen_ai.operation.name"], "retrieve")
            self.assertEqual(attrs["bedrock.knowledge_base.id"], "kb-12345")
            self.assertEqual(attrs["bedrock.retrieval.query"], "company benefits policy")
            self.assertEqual(attrs["bedrock.retrieval.number_of_results"], 5)
            self.assertEqual(attrs["bedrock.retrieval.search_type"], "HYBRID")

    def test_extract_retrieve_and_generate_attributes(self):
        """Test extraction of retrieve_and_generate attributes."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create request dict
            request_dict = {
                "body": json.dumps(
                    {
                        "input": {"text": "What are the company values?"},
                        "sessionId": "rag-session-123",
                        "retrieveAndGenerateConfiguration": {
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseConfiguration": {
                                "knowledgeBaseId": "kb-67890",
                                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2",
                                "generationConfiguration": {
                                    "inferenceConfig": {
                                        "temperature": 0.7,
                                        "maxTokens": 500,
                                        "topP": 0.9,
                                    }
                                },
                            },
                        },
                    }
                )
            }

            attrs = instrumentor._extract_retrieve_and_generate_attributes(request_dict)

            # Assert
            self.assertEqual(attrs["gen_ai.system"], "bedrock_agents")
            self.assertEqual(attrs["gen_ai.operation.name"], "retrieve_and_generate")
            self.assertEqual(attrs["bedrock.rag.input_text"], "What are the company values?")
            self.assertEqual(attrs["bedrock.rag.session_id"], "rag-session-123")
            self.assertEqual(attrs["bedrock.rag.type"], "KNOWLEDGE_BASE")
            self.assertEqual(attrs["bedrock.knowledge_base.id"], "kb-67890")
            self.assertIn("anthropic.claude-v2", attrs["gen_ai.request.model"])
            self.assertEqual(attrs["gen_ai.request.temperature"], 0.7)
            self.assertEqual(attrs["gen_ai.request.max_tokens"], 500)
            self.assertEqual(attrs["gen_ai.request.top_p"], 0.9)

    def test_extract_usage_returns_none(self):
        """Test that _extract_usage returns None (usage tracked by Bedrock model)."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Token usage not exposed by Bedrock Agents
            result = {"sessionId": "test", "completion": "response"}

            usage = instrumentor._extract_usage(result)

            self.assertIsNone(usage)

    def test_extract_response_attributes_invoke_agent(self):
        """Test extraction of response attributes for invoke_agent."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create mock result
            result = {
                "sessionId": "response-session-123",
                "contentType": "application/json",
                "completion": True,
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["bedrock.agent.response.session_id"], "response-session-123")
            self.assertEqual(attrs["bedrock.agent.response.content_type"], "application/json")
            self.assertTrue(attrs["bedrock.agent.response.has_completion"])

    def test_extract_response_attributes_retrieve(self):
        """Test extraction of response attributes for retrieve."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create mock result with retrieval results
            result = {
                "retrievalResults": [
                    {"content": "doc1"},
                    {"content": "doc2"},
                    {"content": "doc3"},
                ]
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertEqual(attrs["bedrock.retrieval.results_count"], 3)

    def test_extract_response_attributes_retrieve_and_generate(self):
        """Test extraction of response attributes for retrieve_and_generate."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create mock result with output and citations
            result = {
                "output": {
                    "text": "The company values are integrity, innovation, and collaboration."
                },
                "citations": [{"reference": "policy_doc_1"}, {"reference": "handbook_2"}],
            }

            attrs = instrumentor._extract_response_attributes(result)

            # Assert
            self.assertIn("integrity", attrs["bedrock.rag.output_text"])
            self.assertEqual(attrs["bedrock.rag.citations_count"], 2)

    def test_extract_finish_reason_with_stop_reason(self):
        """Test extraction of finish reason with stopReason."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create mock result
            result = {"stopReason": "end_turn"}

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "end_turn")

    def test_extract_finish_reason_completed(self):
        """Test extraction of finish reason when completed."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Create mock result with output
            result = {"output": {"text": "response"}}

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertEqual(finish_reason, "completed")

    def test_extract_finish_reason_none(self):
        """Test extraction of finish reason returns None when not available."""
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["bedrock-agent-runtime"]
        mock_boto3.session.Session.return_value = mock_session

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            instrumentor = BedrockAgentsInstrumentor()

            # Empty result
            result = {}

            finish_reason = instrumentor._extract_finish_reason(result)

            # Assert
            self.assertIsNone(finish_reason)


if __name__ == "__main__":
    unittest.main()
