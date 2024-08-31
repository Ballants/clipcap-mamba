from torch import nn
from transformers import MambaForCausalLM, AutoTokenizer


class LLM(nn.Module):
    """
    Processes embedding into caption.
    """

    def __init__(self, model):
        super(LLM, self).__init__()

        self.model = MambaForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.model.config.vocab_size

        # self.freeze_llm()

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(
            inputs_embeds=embedding, attention_mask=attention_mask
        )
        return text_features.logits

    def freeze_llm(self):
        """
        Freezes the parameters of the backbone of the language model, so they are not updated during training.
        """
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.lm_head.parameters():
            p.requires_grad = True
