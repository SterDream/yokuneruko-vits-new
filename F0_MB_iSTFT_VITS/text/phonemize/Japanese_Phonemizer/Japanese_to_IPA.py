import pyopenjtalk

JTALK_TO_IPA = {
    # 母音
    "u": "ɯ",
    # 子音
    "ky": "kʲ", "gy": "gʲ", "ny": "ɲ", "y": "j",
    "sh": "ɕ", "ch": "t͡ɕ", "ts": "t͡s",
    "j": "d͡ʑ", "f": "ɸ", "my": "mʲ",
    "r": "ɾ", "ry": "ɾʲ",
}

class G2P_Japanese_to_Phoneme:
    def g2p(text):
        ipa = []
        text = pyopenjtalk.g2p(text)
        for t in text:
            ipa.append(JTALK_TO_IPA.get(t, t))
        return "".join(ipa).lower().replace(" ", "")


if __name__ == "__main__":
    text = G2P_Japanese_to_Phoneme.g2p("私はアメリカ人です。")
    print(text)

