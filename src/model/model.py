import torch
from transformers import RobertaForSequenceClassification

class VADRobertaModel(torch.nn.Module):
    def __init__(self, model_name: str, dropout_rate: float = 0.1):
        super(VADRobertaModel, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Base model has single output
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )

        # Add separate heads for each dimension
        hidden_size = self.roberta.config.hidden_size
        self.valence_head = torch.nn.Linear(hidden_size, 1)
        self.arousal_head = torch.nn.Linear(hidden_size, 1)
        self.dominance_head = torch.nn.Linear(hidden_size, 1)

        # Initialize heads
        torch.nn.init.normal_(self.valence_head.weight, std=0.02)
        torch.nn.init.normal_(self.arousal_head.weight, std=0.02)
        torch.nn.init.normal_(self.dominance_head.weight, std=0.02)
        torch.nn.init.zeros_(self.valence_head.bias)
        torch.nn.init.zeros_(self.arousal_head.bias)
        torch.nn.init.zeros_(self.dominance_head.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

        valence = self.valence_head(pooled_output)
        arousal = self.arousal_head(pooled_output)
        dominance = self.dominance_head(pooled_output)

        logits = torch.cat([valence, arousal, dominance], dim=1)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }