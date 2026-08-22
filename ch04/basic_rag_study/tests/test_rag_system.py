"""RAG 시스템 테스트.

대부분은 가짜 OpenAI 클라이언트로 돌아 키도 네트워크도 필요 없다.
실제 임베딩 API를 쓰는 것들만 아래쪽에 integration으로 모아뒀다.
"""

from pathlib import Path

import pytest


class TestSplitText:
    def test_short_text_becomes_one_chunk(
        self,
        rag_system,
    ):
        chunks = rag_system._split_text("짧은 문장 하나.")

        assert chunks == ["짧은 문장 하나."]

    def test_long_text_is_split_by_token_count(
        self,
        rag_system,
    ):
        chunks = rag_system._split_text("word " * 3000)

        assert len(chunks) > 1
        for chunk in chunks:
            assert (
                len(rag_system.encoding.encode(chunk)) <= rag_system.settings.chunk_size
            )

    def test_chunks_overlap_so_context_is_not_cut(
        self,
        rag_system,
    ):
        # 전진 폭은 chunk_size - chunk_overlap이므로 청크 수가 그만큼 늘어난다
        tokens = rag_system.encoding.encode("word " * 3000)
        stride = rag_system.settings.chunk_size - rag_system.settings.chunk_overlap
        expected = len(range(0, len(tokens), stride))

        assert len(rag_system._split_text("word " * 3000)) == expected

    def test_blank_text_produces_no_chunks(
        self,
        rag_system,
    ):
        assert rag_system._split_text("   \n\t  ") == []

    def test_chunks_are_stripped(
        self,
        rag_system,
    ):
        for chunk in rag_system._split_text("  앞뒤 공백이 있는 글  "):
            assert chunk == chunk.strip()


class TestCosineSimilarity:
    @pytest.mark.parametrize(
        ("vec", "expected"),
        [
            ([1.0, 0.0, 0.0], 1.0),
            ([0.0, 1.0, 0.0], 0.0),
            ([-1.0, 0.0, 0.0], -1.0),
        ],
    )
    def test_known_angles(
        self,
        rag_system,
        vec,
        expected,
    ):
        assert rag_system.cosine_similarity([1.0, 0.0, 0.0], vec) == pytest.approx(
            expected,
        )

    def test_scale_does_not_matter(
        self,
        rag_system,
    ):
        assert rag_system.cosine_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(
            1.0,
        )


class TestLoadPdfs:
    def test_reads_the_knowledge_folder(
        self,
        rag_system,
    ):
        if not list(Path(rag_system.settings.knowledge_folder).glob("*.pdf")):
            pytest.skip("knowledge_files에 PDF가 없다")

        documents = rag_system.load_pdfs()

        assert documents
        for doc in documents:
            assert set(doc) == {"content", "source", "file_path", "chunk_id"}
            assert doc["content"].strip()
            assert doc["source"].endswith(".pdf")
            assert isinstance(doc["chunk_id"], int)

    def test_chunk_ids_restart_per_file(
        self,
        rag_system,
    ):
        if not list(Path(rag_system.settings.knowledge_folder).glob("*.pdf")):
            pytest.skip("knowledge_files에 PDF가 없다")

        documents = rag_system.load_pdfs()
        first_ids = [d["chunk_id"] for d in documents if d["source"] == documents[0]["source"]]

        assert first_ids == list(range(len(first_ids)))

    def test_empty_folder_yields_nothing(
        self,
        rag_system,
        tmp_path,
    ):
        assert rag_system.load_pdfs(str(tmp_path)) == []

    def test_a_broken_pdf_does_not_stop_the_others(
        self,
        rag_system,
        tmp_path,
    ):
        (tmp_path / "broken.pdf").write_text("이건 PDF가 아니다")

        # 예외를 밖으로 던지지 않고 로그만 남기고 넘어간다
        assert rag_system.load_pdfs(str(tmp_path)) == []


class TestBuildVectorDb:
    def test_documents_embeddings_and_metadata_stay_aligned(
        self,
        rag_system,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(
            rag_system,
            "load_pdfs",
            lambda: [
                {"content": "가", "source": "a.pdf", "file_path": "a", "chunk_id": 0},
                {"content": "나", "source": "a.pdf", "file_path": "a", "chunk_id": 1},
            ],
        )

        rag_system.build_vector_db()

        assert len(rag_system.documents) == 2
        assert len(rag_system.embeddings) == 2
        assert rag_system.metadata == [
            {"source": "a.pdf", "chunk_id": 0},
            {"source": "a.pdf", "chunk_id": 1},
        ]

    def test_second_build_is_skipped(
        self,
        built_rag_system,
        fake_openai,
    ):
        built_rag_system.build_vector_db()

        assert fake_openai.embedding_calls == []

    def test_force_rebuild_runs_again(
        self,
        built_rag_system,
        fake_openai,
        monkeypatch,
    ):
        monkeypatch.setattr(
            built_rag_system,
            "load_pdfs",
            lambda: [{"content": "새 글", "source": "b.pdf", "file_path": "b", "chunk_id": 0}],
        )

        built_rag_system.build_vector_db(force_rebuild=True)

        assert fake_openai.embedding_calls == [["새 글"]]
        assert len(built_rag_system.documents) == 1

    def test_no_documents_leaves_the_database_empty(
        self,
        rag_system,
        monkeypatch,
    ):
        monkeypatch.setattr(rag_system, "load_pdfs", list)

        rag_system.build_vector_db()

        assert rag_system.documents == []


class TestSearch:
    def test_searching_before_building_raises(
        self,
        rag_system,
    ):
        with pytest.raises(ValueError, match="Vector database not built"):
            rag_system.search("무엇이든")

    def test_returns_k_results_with_the_expected_shape(
        self,
        built_rag_system,
    ):
        results = built_rag_system.search("paging", k=2)

        assert len(results) == 2
        for result in results:
            assert set(result) == {"content", "metadata", "score"}
            assert set(result["metadata"]) == {"source", "chunk_id"}
            assert isinstance(result["score"], float)

    def test_scores_are_sorted_high_to_low(
        self,
        built_rag_system,
    ):
        scores = [r["score"] for r in built_rag_system.search("paging", k=3)]

        assert scores == sorted(scores, reverse=True)

    def test_asking_for_more_than_exists_returns_what_there_is(
        self,
        built_rag_system,
    ):
        assert len(built_rag_system.search("paging", k=99)) == 3

    def test_score_belongs_to_the_document_next_to_it(
        self,
        built_rag_system,
    ):
        """정렬한 뒤 점수와 문서를 다른 첨자로 꺼내면 짝이 어긋난다."""
        results = built_rag_system.search("paging", k=3)

        for result in results:
            index = next(
                i
                for i, doc in enumerate(built_rag_system.documents)
                if doc["content"] == result["content"]
            )
            expected = built_rag_system.cosine_similarity(
                built_rag_system.get_embeddings(["paging"])[0],
                built_rag_system.embeddings[index],
            )
            assert result["score"] == pytest.approx(expected)


class TestGetContextForQuery:
    def test_context_names_every_source(
        self,
        built_rag_system,
    ):
        context = built_rag_system.get_context_for_query("paging", k=3)

        for doc in built_rag_system.documents:
            assert doc["source"] in context
            assert doc["content"] in context

    def test_documents_are_numbered_from_one(
        self,
        built_rag_system,
    ):
        context = built_rag_system.get_context_for_query("paging", k=2)

        assert "Document 1 (Source:" in context
        assert "Document 2 (Source:" in context

    def test_context_without_a_database_raises(
        self,
        rag_system,
    ):
        """search()가 먼저 막으므로 빈 문자열이 아니라 예외가 나온다."""
        with pytest.raises(ValueError, match="Vector database not built"):
            rag_system.get_context_for_query("무엇이든")


@pytest.mark.integration
class TestAgainstTheRealApi:
    """실제 OpenAI API를 호출한다. `pytest -m integration`으로만 돈다."""

    def test_embeddings_come_back_with_a_consistent_dimension(
        self,
        real_container,
    ):
        rag = real_container.rag_system()

        embeddings = rag.get_embeddings(["첫 문장", "두 번째 문장"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == len(embeddings[1]) > 0

    def test_similar_text_scores_higher_than_unrelated_text(
        self,
        real_container,
    ):
        rag = real_container.rag_system()

        query, close, far = rag.get_embeddings(
            ["machine learning", "deep learning models", "banana bread recipe"],
        )

        assert rag.cosine_similarity(query, close) > rag.cosine_similarity(query, far)

    def test_chat_completions_are_reachable(
        self,
        real_container,
    ):
        """임베딩과 응답 생성은 별개 엔드포인트라 따로 확인해야 한다."""
        answer = real_container.llm_manager().generate_response(
            "Reply with the single word OK.", max_tokens=5,
        )

        # 실패해도 예외 대신 문자열이 오므로 내용을 봐야 한다
        assert "Error generating response" not in answer
        assert answer.strip()

    def test_full_workflow(
        self,
        real_container,
    ):
        rag = real_container.rag_system()
        if not list(Path(rag.settings.knowledge_folder).glob("*.pdf")):
            pytest.skip("knowledge_files에 PDF가 없다")

        rag.build_vector_db()
        assert len(rag.documents) == len(rag.embeddings) > 0

        results = rag.search("paging", k=3)
        assert len(results) == 3

        context = rag.get_context_for_query("paging", k=2)
        assert "Document 1 (Source:" in context
