import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class GenerativeBart:
    def __init__(self, model_name: str, max_length: int, device: str):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained(model_name)
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
        self, inputs: torch.Tensor, max_length: int = 20
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Return a (sequence, scores) tuple where sequence is a tensor of shape (generation_len)
        containing the ids of the generated sequence, and scores is a list of len generation_len,
        whose each element is a tensor of shape [1, vocab_size] containing the predicted token
        logits for each step.

        """
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

    def decode_prefixes(self, generated_ids: list[torch.Tensor]) -> list[list[str]]:
        results: list[list[str]] = []
        for sequence_ind in range(len(generated_ids)):
            results.append([])
            previous_decoded_length = -1
            for prefix_length in range(len(generated_ids[sequence_ind])):
                prefix = self.tokenizer.batch_decode(
                    [generated_ids[sequence_ind][: prefix_length + 1]], skip_special_tokens=True
                )[0]
                if len(prefix) == previous_decoded_length:
                    break
                    # Break when decoded sequence length stops increasing - otherwise
                    # we'd be decoding the padding
                if len(prefix) != 0:
                    previous_decoded_length = len(prefix)
                    results[-1].append(prefix)
        return results