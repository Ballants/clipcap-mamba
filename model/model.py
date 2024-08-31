import torch
import torch.nn as nn
import torch.nn.functional as F

from lang_model import LLM
from mapping_nets import MambaMapping


class Captioner(nn.Module):
    """
    Final Model class: Mapping network + LLM.
    """

    def __init__(self, llm_model, d_clip=512, prefix_len=10, num_layers=8, device='cpu', embed_size=768):
        """
        Model constructor.
        Args:
            llm_model: name of the llm model to use
            d_clip: dimension of the clip embedding
            prefix_len: length of the prefix
            embed_size: size of the llm embedding
            num_layers: number of Mamba layers in the mapping network
        """
        super(Captioner, self).__init__()

        self.d_clip = d_clip
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.prefix_len = prefix_len
        self.device = device

        self.mapping = MambaMapping(d_clip=self.d_clip, embed_size=self.embed_size, prefix_len=self.prefix_len,
                                    num_mamba_layers=self.num_layers, device=self.device)

        self.llm = LLM(model=llm_model)

    def forward(self, clip_emb, target_cap=None, attention_mask=None, mode=None, max_len=20, temperature=1.0,
                beam_width=5):

        # training and validation
        if target_cap is not None and attention_mask is not None:
            x, x_mask = target_cap[:, :-1], attention_mask[:, :-1]
            y = target_cap[:, 1:]

            img_mapped = self.mapping(clip_emb)  # batch_size, prefix_len, embed_size

            target_llm_emb = self.llm.model.backbone.embeddings(x)  # batch_size, -1 , embed_size

            x = torch.cat([img_mapped, target_llm_emb], dim=1)  # batch_size, -1 , embed_size

            x_mask = torch.cat([torch.ones(x_mask.shape[0], self.prefix_len).to(self.device), x_mask],
                               dim=1)  # batch_size, -1

            pred = self.llm(x, attention_mask=x_mask)  # batch_size, -1, vocab_size

            loss = nn.functional.cross_entropy(
                pred[:, self.prefix_len - 1: -1].reshape(-1, pred.shape[-1]), y.reshape(-1),
                ignore_index=0
            )

            return loss

        # test and prediction
        else:
            if mode == 'greedy':
                tokens = self.greedy_decode(clip_emb, max_len, temperature)
            elif mode == 'beam':
                tokens = self.beam_search_decode(clip_emb, max_len, beam_width, temperature)
            else:
                raise ValueError("Invalid mode. Choose either 'greedy' or 'beam'.")

            decoded = self.llm.tokenizer.decode(tokens[:-1])

            decoded = decoded.strip()
            decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens

    def greedy_decode(self, clip_emb, max_len, temperature=1.0, top_k=0, top_p=0.9):

        with torch.no_grad():
            img_mapped = self.mapping(clip_emb)  # batch_size, prefix_len, d_model

            sos_emb = self.llm.model.backbone.embeddings(
                torch.tensor(self.llm.tokenizer.bos_token_id).to(self.device)
            )
            sos_emb = sos_emb.unsqueeze(0).unsqueeze(0)  # 1, 1, embed_size

            input_ids = torch.cat([sos_emb, img_mapped], dim=1)  # 1, prefix_len+1, embed_size

            tokens = []
            generated = input_ids
            for _ in range(max_len):

                logits = self.llm(generated)  # 1, prefix_len+n, vocab_size

                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits = mask.scatter(1, top_k_indices, top_k_logits)

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = float('-inf')

                    next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1),
                                                   num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)
                tokens.append(next_token.item())

                if next_token == self.llm.tokenizer.eos_token_id:
                    break

        return tokens

    def beam_search_decode(self, clip_emb, max_len, beam_width=5, temperature=1.0):
        with torch.no_grad():
            img_mapped = self.mapping(clip_emb)  # batch_size, prefix_len, d_model

            sos_token_id = self.llm.tokenizer.bos_token_id
            sos_emb = self.llm.model.backbone.embeddings(
                torch.tensor([sos_token_id]).to(self.device)
            )
            sos_emb = sos_emb.unsqueeze(0)  # 1, 1, embed_size

            start_emb = torch.cat([img_mapped, sos_emb], dim=1)  # 1, prefix_len+1, embed_size

            beams = [([], 0.0, start_emb)]  # List of (tokens, score)

            for _ in range(max_len):
                all_candidates = []
                for seq, score, emb in beams:
                    if len(seq) and seq[-1] == self.llm.tokenizer.eos_token_id:
                        all_candidates.append((seq, score, emb))
                        continue

                    pred = self.llm(emb)  # 1, prefix_len+n, vocab_size
                    pred = F.log_softmax(pred[:, -1, :] / temperature, dim=-1)  # 1, vocab_size

                    top_k_scores, top_k_tokens = torch.topk(pred, beam_width, dim=-1)  # 1, beam_width

                    # Create new candidate beams
                    for i in range(beam_width):
                        candidate_score = score + (top_k_scores[0, i].item())
                        next_token = top_k_tokens[0, i].item()
                        candidate_seq = seq + [next_token]
                        tok_emb = self.llm.model.backbone.embeddings(
                            torch.tensor(top_k_tokens[0, i].item()).unsqueeze(0).unsqueeze(0).to(self.device)
                        )
                        candidate_emb = torch.cat([emb, tok_emb], dim=1)
                        all_candidates.append((candidate_seq, candidate_score, candidate_emb))

                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:beam_width]

            return beams[0][0]  # Return the best sequence
