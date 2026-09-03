from pathlib import Path

from tradingagents.scheduler.actions import SchedulerAction


class FakeTokenizer:
    def __init__(self, base_vocab_size: int = 64):
        self.base_vocab_size = base_vocab_size
        self.vocab = {f"token-{index}": index for index in range(base_vocab_size)}
        self.pad_token_id = None
        self.eos_token_id = 1
        self.eos_token = "<EOS>"
        self._pad_token = None

    def __len__(self) -> int:
        return len(self.vocab)

    def add_special_tokens(self, values) -> int:
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.eos_token_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def __call__(self, text: str, *, add_special_tokens: bool = True):
        if text in self.vocab:
            return {"input_ids": [self.vocab[text]]}
        values = [2 + (ord(character) % max(1, self.base_vocab_size - 2)) for character in text]
        return {"input_ids": values or [2]}

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        return [2 + (ord(character) % max(1, self.base_vocab_size - 2)) for character in text]

    def save_pretrained(self, path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "fake_tokenizer.txt").write_text("test-only", encoding="utf-8")

    @property
    def action_ids(self) -> dict[SchedulerAction, int]:
        return {action: self.vocab[action.value] for action in SchedulerAction}
