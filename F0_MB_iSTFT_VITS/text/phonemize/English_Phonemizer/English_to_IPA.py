from espeak_phonemizer import Phonemizer
import re

class G2P_English_to_Phoneme:
    def __init__(self, default_voice: str="en-us"):
        self.phonemizer = Phonemizer(default_voice=default_voice)

    def __call__(self, text):
        text = self.phonemizer.phonemize(text.replace(" ", ""), keep_clause_breakers=True)
        text = re.sub(r"[.,!?;。、「」！？；…—～\-\s]", 'pau', text)
        text = text.replace("paupau", "pau")
        return text


if __name__ == "__main__":
    phonemizer = G2P_English_to_Phoneme("en")
    text = phonemizer("my english is terrible. my first language is not english.")
    print(text)
