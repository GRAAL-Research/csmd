import torch
import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import utils_sampling

"""
A generator class using a model card. Based on Keep It Simple codebase https://github.com/tingofurro/keep_it_simple.
"""


class Generator:
    def __init__(
        self,
        model_card,
        max_input_length=300,
        max_output_length=25,
        device="cuda",
    ):
        # `model_card` can be a pretrained model name, or a model_folder
        self.model_card = model_card

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_card)
        self.model = GPT2LMHeadModel.from_pretrained(model_card)
        self.tokenizer.pad_token = "!"

        self.start_id = self.tokenizer.bos_token_id

        self.model.to(device)
        self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def generate(self, bodies, max_batch_size=8, progress=False, num_runs=1, **kwargs):
        self.model.eval()

        N_start = len(bodies)
        if num_runs > 1:
            bodies = [body for body in bodies for i in range(num_runs)]
        N = len(bodies)

        outputs = []
        iterator = range(0, N, max_batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            batch_bodies = bodies[i : min(N, i + max_batch_size)]
            with torch.no_grad():
                batch_outputs = self.generate_batch(batch_bodies, **kwargs)
            outputs += batch_outputs

        if num_runs > 1:
            # Refold the number of runs into N outputs
            final_outputs = []
            for i in range(N_start):
                all_runs = outputs[num_runs * i : (num_runs * (i + 1))]
                final_outputs.append(all_runs)
            outputs = final_outputs
        return outputs

    def preprocess_input(self, texts):
        tokenizer_outs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding="longest"
        )
        encs = tokenizer_outs["input_ids"]
        attention_mask = tokenizer_outs["attention_mask"]

        encs = encs[:, : self.max_input_length].to(self.device)
        attention_mask = attention_mask[:, : self.max_input_length].to(self.device)
        return encs, attention_mask

    def encode(self, encoded_texts):
        input_ids, attention_mask = encoded_texts

        model_outs = self.model(input_ids=input_ids, past_key_values=None)
        return model_outs["past_key_values"]

    def decode_fast(self, decoded_so_far, past):
        model_outputs = self.model(
            input_ids=decoded_so_far[:, -1].view(-1, 1), past_key_values=past
        )
        return model_outputs["logits"], model_outputs["past_key_values"]

    def toks2text_batch(self, tokens_batch, return_tokens=False):
        end_id = self.tokenizer.eos_token_id

        tokens_batch = [
            tokens[1:].tolist() + [end_id] for tokens in tokens_batch
        ]  # Add the end_id just in case
        tokens_batch = [
            tokens[: tokens.index(end_id)] for tokens in tokens_batch
        ]  # Cut at the end token

        texts = self.tokenizer.batch_decode(
            tokens_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        if not return_tokens:
            return texts
        else:
            return texts, tokens_batch

    def generate_batch(
        self,
        encoded_texts,
        max_output_length=100,
        sample=False,
        force_start=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        no_copy_ngram=0,
        no_repeat_ngram=0,
        min_length=0,
    ):
        N = len(encoded_texts)

        force_start_ids = []
        if force_start is not None:
            force_start_ids = self.tokenizer.encode(
                force_start, add_special_tokens=False
            )

        if self.model_card == "facebook/bart-large-cnn":
            force_start_ids = [0]

        inputs = self.preprocess_input(encoded_texts)
        past = self.encode(inputs)

        build_up = torch.LongTensor([self.start_id]).repeat(N, 1).to(self.device)

        seq_logprobs = torch.zeros((N)).to(self.device)

        end_id = self.tokenizer.eos_token_id
        finished_func = lambda build_up: all(
            [end_id in build for build in build_up[:, 1:]]
        )

        while build_up.shape[1] <= max_output_length and not finished_func(build_up):
            is_force_start = len(force_start_ids) > 0 and build_up.shape[1] <= len(
                force_start_ids
            )

            logits, past = self.decode_fast(build_up, past)

            logits = logits.view(N, -1)

            logits = utils_sampling.ngram_copy_filtering(
                build_up, inputs[0], logits, n_gram=no_copy_ngram
            )
            logits = utils_sampling.ngram_copy_filtering(
                build_up, build_up, logits, n_gram=no_repeat_ngram
            )
            logits = utils_sampling.top_k_top_p_filtering(
                logits, top_k=top_k, top_p=top_p
            )

            if (
                min_length > 0
                and build_up.shape[1] <= min_length
                and not is_force_start
            ):
                logits[:, end_id] -= 1000.0

            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

            if is_force_start:
                force_idx = build_up.shape[1] - 1
                current = (
                    torch.LongTensor([force_start_ids[force_idx]])
                    .repeat(N, 1)
                    .to(self.device)
                )
            elif sample:
                probs = torch.nn.functional.softmax(
                    logits / temperature, dim=-1
                ).squeeze(1)
                distrib = torch.distributions.categorical.Categorical(probs)
                current = distrib.sample().unsqueeze(-1)
            else:
                current = torch.argmax(logprobs, dim=-1)

            current = current.view(-1, 1)
            build_up = torch.cat((build_up, current), dim=1)

            not_finished = (1 - torch.any(build_up[:, 1:] == end_id, dim=1).float()).to(
                self.device
            )
            if not (
                self.model_card == "facebook/bart-large-cnn" and is_force_start
            ):  # otherwise we force pick an end token at the start
                seq_logprobs += not_finished * logprobs[
                    torch.arange(N), current.view(N)
                ].view(N)

        outputs = {}
        outputs["output_text"], outputs["output_tokens"] = self.toks2text_batch(
            build_up, return_tokens=True
        )
        outputs["logprob"] = seq_logprobs.tolist()

        outputs_list = [{k: outputs[k][i] for k in outputs} for i in range(N)]
        return outputs_list
