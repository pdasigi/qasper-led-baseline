import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable, Tuple

from overrides import overrides

import spacy
import torch

from allennlp.common.util import JsonDict
from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
    TensorField,
)
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("qasper")
class QasperReader(DatasetReader):
    """
    Reads a JSON-formatted Qasper data file and returns a `Dataset` where the `Instances` have
    four fields:
     * `question_with_context`, a `TextField` that contains the concatenation of question and
       context,
     * `paragraph_indices`, a `ListField` of `IndexFields` indicating paragraph-start tokens
       in `question_with_context`.
     * `global_attention_mask`, a mask that can be used by a longformer to specify which tokens in
       `question_with_context` should have global attention (only present if
       `include_global_attention_mask` is `True`).
     * `evidence`, a 0/1 `TensorField` indicating whether each paragraph in `paragraph_indices`
       should be selected as evidence.
     * `answer`, a `TextField` that contains the (wordpiece-tokenized) answer to the question
     * A `MetadataField` that stores the instance's ID, paper ID, the original question, the
       original passage text, both of these in tokenized form, and the context also broken into
       paragraphs, and the gold evidence spans, accessible as `metadata['question_id']`,
       `metadata['article_id']`, `metadata['question']`, `metadata['context']`,
       `metadata['question_tokens']`, `metadata['context_tokens']`,
       `metadata['context_paragraphs']`, `metadata['all_evidence']`, `metadata['all_answers']`.

    Parameters
    ----------
    transformer_model_name : `str`, optional (default=`allenai/led-large-16384`)
        This reader chooses tokenizer and token indexer according to this setting.
    max_query_length : `int`, optional (default=128)
        The maximum number of wordpieces dedicated to the question. If the question is longer than
        this, it will be truncated.
    max_document_length : `int` , optional (default=16384)
        This is the maximum number of wordpieces allowed per one whole document (including the
        question, for simplicity).  If the document is longer than this many word pieces, it will be
        truncated.
    paragraph_separator : `Optional[str]`, optional (default="</s>")
        If given, we will use this as a separator token in between paragraphs.  Pass in `None` to
        have this not be used.
    include_global_attention_mask : `bool` (default = True)
        If `True`, we will include a field in the output containing a global attention mask for use
        with a longformer, which is `True` for all starts of paragraphs and question tokens, so
        attention will always be placed on those tokens.
    for_training : `bool` (default = False)
        This flag affects how questions with multiple answers are handled. When set to True, this flag
        causes the reader to yield one instance per answer. When set to False, the instance will contain
        only the first answer. The metadata will always contain all the answers and evidence, which can be
        used at evaluation time to compute aggregated metrics.
    """

    def __init__(
        self,
        transformer_model_name: str = "allenai/led-large-16384",
        max_query_length: int = 128,
        max_document_length: int = 16384,
        paragraph_separator: Optional[str] = "</s>",
        include_global_attention_mask: bool = True,
        for_training: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._transformer_model_name = transformer_model_name
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._include_global_attention_mask = include_global_attention_mask
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)
        }
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self._paragraph_separator = paragraph_separator
        self._for_training = for_training
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset")
        if file_path.endswith(".json"):
            yield from self._read_json(file_path)
        elif file_path.endswith(".jsonl"):
            yield from self._read_json_lines(file_path)
        else:
            raise RuntimeError(
                f"Unsupported extension on file: {file_path}. Only json and jsonl are supported."
            )

    def _read_json(self, file_path: str):
        logger.info("Reading json file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        for article_id, article in self.shard_iterable(dataset.items()):
            article["article_id"] = article_id
            yield from self._article_to_instances(article)
        self._log_stats()

    def _read_json_lines(self, file_path: str):
        logger.info("Reading json lines file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            for data_line in self.shard_iterable(dataset_file):
                data = json.loads(data_line)
                yield from self._article_to_instances(data)
        self._log_stats()

    def _log_stats(self) -> None:
        logger.info("Stats:")
        for key, value in self._stats.items():
            logger.info("%s: %d", key, value)

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_full_text(article["full_text"])
        tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
            paragraphs
        )

        self._stats["number of documents"] += 1
        for question_answer in article["qas"]:
            self._stats["number of questions"] += 1
            self._stats["number of answers"] += len(question_answer["answers"])
            if len(question_answer["answers"]) > 1:
                self._stats["questions with multiple answers"] += 1

            all_answers = []
            all_evidence = []
            all_evidence_masks = []
            for answer_annotation in question_answer["answers"]:
                answer, evidence = self._extract_answer_and_evidence(
                    question_answer["answers"][0]["answer"]
                )
                all_answers.append(answer)
                all_evidence.append(evidence)
                evidence_mask = self._get_evidence_mask(evidence, paragraphs)
                all_evidence_masks.append(evidence_mask)

            additional_metadata = {
                "question_id": question_answer["question_id"],
                "article_id": article.get("article_id"),
                "all_answers": all_answers,
                "all_evidence": all_evidence,
                "all_evidence_masks": all_evidence_masks,
            }
            answers_to_yield = all_answers if self._for_training else [all_answers[0]]
            evidence_masks_to_yield = all_evidence_masks if self._for_training else [all_evidence_masks[0]]
            for answer, evidence_mask in zip(answers_to_yield, evidence_masks_to_yield):
                yield self.text_to_instance(
                    question_answer["question"],
                    paragraphs,
                    tokenized_context,
                    paragraph_start_indices,
                    evidence_mask,
                    answer,
                    additional_metadata,
                )

    @staticmethod
    def _get_evidence_mask(evidence: List[str], paragraphs: List[str]) -> List[int]:
        """
        Takes a list of evidence snippets, and the list of all the paragraphs from the
        paper, and returns a list of indices of the paragraphs that contain the evidence.
        """
        if not evidence:
            return []
        evidence_mask = []
        for i, paragraph in enumerate(paragraphs):
            for evidence_str in evidence:
                if evidence_str in paragraph:
                    evidence_mask.append(1)
                    break
            else:
                evidence_mask.append(0)

        return evidence_mask

    @overrides
    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        question: str,
        paragraphs: List[str],
        tokenized_context: List[Token] = None,
        paragraph_start_indices: List[int] = None,
        evidence_mask: List[int] = None,
        answer: str = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        tokenized_question = self._tokenizer.tokenize(question)
        if not tokenized_context or not paragraph_start_indices:
            tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                paragraphs
            )

        # make the question field
        question_field = TextField(
            self._tokenizer.add_special_tokens(tokenized_question, tokenized_context),
        )
        fields["question_with_context"] = question_field

        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + len(self._tokenizer.sequence_pair_mid_tokens)
        )

        paragraph_indices_list = [x + start_of_context for x in paragraph_start_indices]

        paragraph_indices_field = ListField(
            [IndexField(x, question_field) for x in paragraph_indices_list]
        )
        fields["paragraph_indices"] = paragraph_indices_field

        if self._include_global_attention_mask:
            # We need to make a global attention array. We'll use all the paragraph indices and the
            # indices of question tokens.
            mask_indices = set(list(range(start_of_context)) + paragraph_indices_list)
            mask = [
                True if i in mask_indices else False for i in range(len(question_field))
            ]
            fields["global_attention_mask"] = TensorField(torch.tensor(mask))

        if evidence_mask is not None:
            evidence_field = TensorField(torch.tensor(evidence_mask))
            fields["evidence"] = evidence_field

        if answer:
            fields["answer"] = TextField(self._tokenizer.tokenize(answer))

        # make the metadata
        metadata = {
            "question": question,
            "question_tokens": tokenized_question,
            "paragraphs": paragraphs,
            "context_tokens": tokenized_context,
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["question_with_context"].token_indexers = self._token_indexers
        instance.fields["answer"].token_indexers = self._token_indexers

    def _tokenize_paragraphs(
        self, paragraphs: List[str]
    ) -> Tuple[List[Token], List[int]]:
        tokenized_context = []
        paragraph_start_indices = []
        for paragraph in paragraphs:
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            paragraph_start_indices.append(len(tokenized_context))
            tokenized_context.extend(tokenized_paragraph)
            if self._paragraph_separator:
                tokenized_context.append(Token(self._paragraph_separator))
        if self._paragraph_separator:
            # We added the separator after every paragraph, so we remove it after the last one.
            tokenized_context = tokenized_context[:-1]
        return tokenized_context, paragraph_start_indices

    def _extract_answer_and_evidence(
        self, answer: List[JsonDict]
    ) -> Tuple[str, List[str]]:
        evidence_spans = [x.replace("\n", " ").strip() for x in answer["evidence"]]
        evidence_spans = [x for x in evidence_spans if x != ""]
        if not evidence_spans:
            self._stats["answers with no evidence"] += 1
        # TODO (pradeep): Deal with figures and tables.
        if any(["FLOAT SELECTED" in span for span in evidence_spans]):
            # Ignoring question if any of the selected evidence is a table or a figure.
            self._stats["answers with table or figure as evidence"] += 1
        if len(evidence_spans) > 1:
            self._stats["multiple_evidence_spans_count"] += 1

        answer_string = None
        if answer.get("unanswerable", False):
            self._stats["unanswerable questions"] += 1
            answer_string = "Unanswerable"
        elif answer.get("yes_no") is not None:
            self._stats["yes/no questions"] += 1
            answer_string = "Yes" if answer["yes_no"] else "No"
        elif answer.get("extractive_spans", []):
            self._stats["extractive questions"] += 1
            if len(answer["extractive_spans"]) > 1:
                self._stats["extractive questions with multiple spans"] += 1
            answer_string = ", ".join(answer["extractive_spans"])
        else:
            answer_string = answer.get("free_form_answer", "")
            if not answer_string:
                self._stats["questions with empty answer"] += 1
            else:
                self._stats["freeform answers"] += 1

        return answer_string, evidence_spans

    @staticmethod
    def _get_paragraphs_from_full_text(full_text: List[JsonDict]) -> List[str]:
        paragraphs = []
        for section_info in full_text:
            # TODO (pradeep): It is possible there are other discrepancies between plain text, LaTeX and HTML.
            # Do a thorough investigation and add tests.
            for paragraph in section_info["paragraphs"]:
                paragraph_text = paragraph.replace("\n", " ").strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
        return paragraphs
