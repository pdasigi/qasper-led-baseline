from typing import Any, Dict, List, Tuple
from overrides import overrides

from transformers import AutoTokenizer
import torch
from torch.nn import MarginRankingLoss

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.training.metrics import Average

from allennlp_models.rc.tools import squad

from qasper_baselines import model, dataset_reader


@Model.register("qasper_calibrator")
class QasperCalibrator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        serialized_model_path: str,
        num_samples: int = 20,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        model_archive = load_archive(serialized_model_path)
        self._qasper_led = model_archive.model.transformer
        self._tokenizer = model_archive.model.tokenizer
        self._num_samples = num_samples
        self._top_answer_f1 = Average()
        self._oracle_answer_f1 = Average()
        self._loss_function = MarginRankingLoss()

    def forward(
        self,
        question_with_context: TextFieldTensors,
        paragraph_indices: torch.Tensor,
        global_attention_mask: torch.Tensor = None,
        evidence: torch.Tensor = None,
        answer: TextFieldTensors = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(question_with_context)
        attention_mask = util.get_text_field_mask(question_with_context)

        generation_output = self._sample_with_grad(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
        )
        # TODO (pradeep): Assuming batch size is 1.
        assert len(metadata) == 1
        predictions, model_scores = self._get_predictions_and_scores(generation_output)
        gold_answers = [a["text"] for a in metadata[0]["all_answers"]]
        f1_scores = [max([squad.compute_f1(sample_prediction, gold_answer) for gold_answer in gold_answers])
                     for sample_prediction in predictions]
        target_list = []
        input_scores_1 = []
        input_scores_2 = []
        for i in range(len(f1_scores) - 1):
            for j in range(i+1, len(f1_scores)):
                input_scores_1.append(model_scores[i])
                input_scores_2.append(model_scores[j])
                target_list.append(1 if f1_scores[i] >= f1_scores[j] else -1)

        scores_i = torch.stack(input_scores_1)
        scores_j = torch.stack(input_scores_2)
        target = torch.tensor(target_list, device=scores_i.device)
        loss = self._loss_function(scores_i, scores_j, target)

        self._oracle_answer_f1(max(f1_scores))
        top_f1_score = sorted(zip([x.tolist() for x in model_scores], f1_scores), key=lambda x: x[0])[-1][1]
        self._top_answer_f1(top_f1_score)

        return {"loss": loss}

    def _sample_with_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor
    ):
        # .generate() in HuggingFace Transformers has torch.no_grad() set. We want the tensors
        # attached to the computation graph. So we'll call the relevant code in .generate() here.
        num_beams = self._qasper_led.config.num_beams
        num_beam_groups = self._qasper_led.config.num_beam_groups
        max_length = self._qasper_led.config.max_length
        pad_token_id = self._qasper_led.config.pad_token_id
        bos_token_id = self._qasper_led.config.bos_token_id
        eos_token_id = self._qasper_led.config.eos_token_id
        model_kwargs = {
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask
        }
        encoder_input_ids = input_ids
        # add encoder_outputs to model_kwargs
        model_kwargs = self._qasper_led._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
        # set input_ids as decoder_input_ids
        input_ids = self._qasper_led._prepare_decoder_input_ids_for_generation(
            input_ids,
            bos_token_id=bos_token_id
        )
        # set model_kwargs
        model_kwargs["use_cache"] = None
        # get distribution pre_processing samplers
        logits_processor = self._qasper_led._get_logits_processor(
            repetition_penalty=None,
            no_repeat_ngram_size=None,
            encoder_no_repeat_ngram_size=None,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=None,
            min_length=None,
            max_length=max_length,
            eos_token_id=None,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=None,
        )

        stopping_criteria = self._qasper_led._get_stopping_criteria(max_length=max_length, max_time=None)

        # get probability distribution warper
        logits_warper = self._qasper_led._get_logits_warper(num_beams=num_beams)

        # expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._qasper_led._expand_inputs_for_generation(
            input_ids,
            expand_size=self._num_samples,
            is_encoder_decoder=True,
            **model_kwargs,
        )

        # sample
        return self._qasper_led.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            **model_kwargs,
        )


    def _get_predictions_and_scores(self, generation_output) -> Tuple[List[str], List[float]]:
        predictions = []
        token_log_probs = []
        normalized_log_probs = []
        output_sequences = generation_output.sequences.tolist()
        output_scores = generation_output.scores
        for answer_id, sequence in enumerate(output_sequences):
            predictions.append(self._tokenizer.decode(sequence, skip_special_tokens=True))
            token_log_probs.append([])
            word_pieces = self._tokenizer.convert_ids_to_tokens(sequence)
            # Skipping the first token id because that is the sep token, and the scores start from the
            # second token.
            for token_id, token, token_scores in zip(sequence[1:], word_pieces[1:], output_scores):
                if token == "<pad>":
                    break
                token_log_prob = torch.log(torch.softmax(token_scores[answer_id], 0)[token_id])
                token_log_probs[-1].append(token_log_prob)
            normalized_log_probs.append(sum(token_log_probs[-1]) / len(token_log_probs[-1]))

        return predictions, normalized_log_probs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "top_f1": self._top_answer_f1.get_metric(reset),
            "oracle_f1": self._oracle_answer_f1.get_metric(reset),
        }
