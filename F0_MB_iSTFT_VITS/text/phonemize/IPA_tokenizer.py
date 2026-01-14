import torch
from .to_IPA_converter import auto_g2p

IPA_BASE = [
    # vowels
    "a", "i", "u", "e", "o", "ɯ", "ʌ", "ɛ", "ɔ", "ɨ", "ə",
    # consonants
    "p", "b", "t", "d", "k", "g", "m", "n", "ŋ",
    "s", "z", "ʃ", "ʂ", "ɕ", "ʑ",
    "h", "ɾ", "l", "r",
    "t͡ɕ", "d͡ʑ",
]

IPA_MODIFIERS = ["ʰ", "ʼ", "ː", "̚"]
IPA_TONES = ["˥", "˦", "˧", "˨", "˩"]
SPECIAL = ["pau", "sil"]

ALL_TOKENS = sorted(
    SPECIAL + IPA_BASE + IPA_MODIFIERS + IPA_TONES,
    key=len,
    reverse=True
)

class IPATokenizer:
    def __init__(self):
        self.tokens = ALL_TOKENS

    def tokenize(self, ipa: str):
        result = []
        i = 0
        while i < len(ipa):
            matched = False
            for tok in self.tokens:
                if ipa.startswith(tok, i):
                    result.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                result.append(ipa[i])
                i += 1
        return result


class PhonemeVocab:
    def __init__(self):
        self.symbol_to_id = {}
        self.id_to_symbol = {}

        self._add("<pad>")
        self._add("<unk>")
        self._add("<bos>")
        self._add("<eos>")
        self._add("pau")
        self._add("sil")

    def _add(self, s):
        if s not in self.symbol_to_id:
            idx = len(self.symbol_to_id)
            self.symbol_to_id[s] = idx
            self.id_to_symbol[idx] = s

    def encode(self, tokens):
        ids = []
        for t in tokens:
            if t not in self.symbol_to_id:
                self._add(t)
            ids.append(self.symbol_to_id.get(t, self.symbol_to_id["<unk>"]))
        return ids

    def decode(self, ids):
        return [self.id_to_symbol.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.symbol_to_id)
    

class Tokenizer:
    def __init__(self):
        self.ipa_tokenizer = IPATokenizer()
        self.vocab = PhonemeVocab()
        self.g2p = auto_g2p()

    def __call__(self, text):
        ipa, _ = self.g2p(text)
        tokens = self.ipa_tokenizer.tokenize("".join(ipa))
        ids = self.vocab.encode(tokens)
        return ids
    
    def to_tensor(self, text):
        ipa, lang = self.g2p(text)
        with torch.no_grad():
            x = torch.LongTensor(ipa)
            x = x.cuda().unsqueeze(0)
            x_lengths = torch.LongTensor([x.size(0)]).cuda()
        return x, x_lengths, lang


if __name__ == "__main__":
    tokenizer = Tokenizer()
    ids = tokenizer("我毎天都喝珍珠奶茶!")
    print(ids)

    print("Vocab size:", len(PhonemeVocab()))
    print("First 10 tokens:", list(PhonemeVocab().symbol_to_id.items())[:10])