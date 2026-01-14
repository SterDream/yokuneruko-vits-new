import re

def English_converter(text):
    from .English_Phonemizer import G2P_English_to_Phoneme
    phonemizer = G2P_English_to_Phoneme(default_voice="en")
    return phonemizer(text)

def Chinese_converter(text):
    from .Chinese_Phonemizer import G2P_Chinese_to_Phoneme
    model = G2P_Chinese_to_Phoneme()
    return model(text)

def Japanese_converter(text):
    from .Japanese_Phonemizer import G2P_Japanese_to_Phoneme
    return G2P_Japanese_to_Phoneme.g2p(text)

def Korean_converter(text):
    from .Korean_Phonemizer import G2P_Korean_to_Phoneme
    g2p = G2P_Korean_to_Phoneme()
    return g2p(text)


RE_EN = re.compile(r"[a-zA-Z]")
RE_ZH = re.compile(r"[一-龯]")
RE_JA = re.compile(r"[ぁ-んァ-ン]")
RE_KO = re.compile(r"[가-힣]")

class auto_g2p:
    def __init__(self):
        self.g2p_map = {
            "en": English_converter,
            "zh": Chinese_converter,
            "ja": Japanese_converter,
            "ko": Korean_converter,
        }

    def __call__(self, text: str):
        lang = self.detect_language(text)
        if lang not in self.g2p_map:
            raise ValueError(f"Unsupported language: {lang}")
        ipa_tokens = self.g2p_map[lang](text)
        return ipa_tokens, lang
   
    def detect_language(self, text: str) -> str:
        if RE_KO.search(text):
            # print("ko")
            return "ko"
        if RE_JA.search(text):
            # rint("ja")
            return "ja"
        if RE_ZH.search(text):
            # print("zh")
            return "zh"
        if RE_EN.search(text):
            # print("en")
            return "en"
        return "unknown"


if __name__ == "__main__":
    print(English_converter("Hello world"))
    print(Chinese_converter("你好，世界"))
    print(Japanese_converter("こんにちは、世界"))
    print(Korean_converter("안녕하세요, 세계"))

    au = auto_g2p()
    ipa_tokens, lang = au("お早うございます。")
    print(ipa_tokens, lang)
    