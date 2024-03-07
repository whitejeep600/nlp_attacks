from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from src.generation.value_model import ValueModel


class GenerativeBart:
    def __init__(
        self, model_name: str, max_length: int, device: str, weights_path: Path | None = None
    ):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
        if weights_path is not None:
            self.bert.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.bert.to(device)
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def parameters(self):
        return self.bert.parameters()

    def generate_with_greedy_decoding(
        self, inputs: torch.Tensor, max_length: int | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if max_length is None:
            max_length = self.max_length
        """
        Return a (sequence, scores) tuple where sequence is a tensor of shape (generation_len)
        containing the ids of the generated sequence, and scores is a list of len generation_len,
        whose each element is a tensor of shape [1, vocab_size] containing the predicted token
        logits for each step.

        """
        inputs = inputs.to(self.device)
        decoded = torch.Tensor([[self.bert.config.decoder_start_token_id]]).int().to(self.device)
        scores: list[torch.Tensor] = []
        for _ in range(max_length - 1):
            next_one = self.bert(
                input_ids=inputs,
                decoder_input_ids=decoded,
            )
            new_scores = next_one.logits[0][-1, :]
            next_id = torch.argmax(new_scores, dim=-1)
            decoded = torch.cat((decoded, torch.Tensor([[next_id]]).int().to(self.device)), dim=-1)
            scores.append(new_scores.unsqueeze(0))
            if next_id == self.bert.generation_config.eos_token_id:
                break
        return decoded.squeeze(0), scores

    def generate_with_random_sampling(
        self, inputs: torch.Tensor, max_length: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length is None:
            max_length = self.max_length
        """
        As above, but with each next token sampled with its predicted probability,
        instead of greedy decoding. Also, return a simple tensor of generation
        probabilities at each step.

        """
        inputs = inputs.to(self.device)
        decoded = torch.Tensor([[self.bert.config.decoder_start_token_id]]).int().to(self.device)
        probabilities: list[torch.Tensor] = []
        for _ in range(max_length - 1):
            next_one = self.bert(
                input_ids=inputs,
                decoder_input_ids=decoded,
            )
            new_scores = next_one.logits[0][-1, :]
            if new_scores.isnan().any().item():
                print(inputs)
                print(decoded)
                exit(2137)
            new_probabilities = torch.softmax(new_scores, dim=-1)
            next_id = torch.multinomial(new_probabilities, 1, replacement=True)[0]
            decoded = torch.cat((decoded, torch.Tensor([[next_id]]).int().to(self.device)), dim=-1)
            probabilities.append(new_probabilities[next_id].reshape(1))
            if next_id == self.bert.generation_config.eos_token_id:
                break
        return decoded.squeeze(0), torch.cat(probabilities, dim=-1)

    def decode(self, generated_ids: list[torch.Tensor]) -> list[str]:
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def decode_prefixes(self, generated_ids: list[torch.Tensor]) -> list[list[str]]:
        results: list[list[str]] = []
        for sequence_ind in range(len(generated_ids)):
            results.append([])
            for prefix_length in range(2, len(generated_ids[sequence_ind]) + 1):
                prefix = self.tokenizer.batch_decode(
                    [generated_ids[sequence_ind][:prefix_length]], skip_special_tokens=True
                )[0]
                results[-1].append(prefix)
        return results

    # Debugging utils
    def token_to_tokenizer_id(self, word: str) -> int:
        return self.tokenizer.encode(word)[1]

    def tokenizer_id_to_token(self, tokenizer_id: int) -> str:
        return self.tokenizer.decode(tokenizer_id)

    # This function was designed for use with the PPO algorithm. The input is two token tensors,
    # with input_ids being the token ids of some sequence, and output_ids being some other
    # GenerativeBart's output for that sequence (for compatibility it is required that the
    # token-id mappins of both models' tokenizers are the same, because I was too lazy to implement
    # it any other way, and I don't see a use-case for that). The output is a tensor of the same
    # length as output_ids, minus one for the START token, containing this model's
    # _probabilities_ for generating the tokens of the output, with the same input. These
    # probabilities are obtained with an algo resembling teacher forcing.
    # The purpose of this is to get the ratios of the generating model's probabilities to these
    # "reference" probabilities, which later gets plugged into PPO's policy loss.
    def get_reference_probabilities(
        self,
        input_ids: torch.Tensor,  # shape: [input_seq_len]
        output_ids: torch.Tensor,  # shape: [output_seq_len]
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        output_ids = output_ids.to(self.device)
        reference_probabilities: list[torch.Tensor] = []
        for i in range(1, len(output_ids)):
            next_one = self.bert(
                input_ids=input_ids.unsqueeze(0),
                decoder_input_ids=output_ids[:i].unsqueeze(0),
            )
            probabilities = torch.softmax(next_one.logits[0][-1], dim=0)
            next_probability = probabilities[output_ids[i]].reshape(1)
            reference_probabilities.append(next_probability)
        return torch.cat(reference_probabilities, dim=0)

    # This interface is as convenient as it gets, but it's mostly useful for debugging.
    def generate_response(
        self,
        input_text: str,
        reference_model: "GenerativeBart",  # no this is really just for debugging
    ) -> str:
        tokenized_text = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"]
        output_ids, logits = self.generate_with_greedy_decoding(input_ids, 16)
        generated_prefixes = self.decode_prefixes([output_ids])[0]
        generated_sequence = generated_prefixes[-1]

        expected_response_ids = input_ids
        expected_prefixes = self.decode_prefixes(expected_response_ids)[0]

        # generated_values = [value_model.get_value(prefix, input_text) for prefix in generated_prefixes]
        # expected_values = [value_model.get_value(prefix, input_text) for prefix in expected_prefixes]
        pass
        # token_probabilites = torch.stack(
        #     [torch.softmax(logits[i][0], dim=0)[output_ids[i + 1]] for i in range(len(logits))],
        # )
        # reference_probabilites = reference_model.get_reference_probabilities(input_ids, output_ids)
        return generated_sequence
