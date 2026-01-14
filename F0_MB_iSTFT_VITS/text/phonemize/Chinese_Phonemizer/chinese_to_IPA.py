from g2pM import G2pM
from dragonmapper.transcriptions import pinyin_to_ipa
from chinese_converter import to_simplified

# tone
TONE_2 = "˧˥"
TONE_3 = "˨˩˦"

PUNCT = set(".,!?;。、，！？；…—～-")

class G2P_Chinese_to_Phoneme:
    def __init__(self):
        self.model = G2pM()

    def __call__(self, text):
        outputs = []
        text = to_simplified(text)
        text = self.model(text, tone=True)

        for t in text:
            # Delete space
            if t.strip() == "":
                continue
            # Special symbol to "pau"
            if t in PUNCT:
                outputs.append("pau")
                continue

            t = pinyin_to_ipa(t).replace("˧˩˧", "˨˩˦")
            outputs.append(t)

        # 三声
        outputs = self.apply_third_tone_sandhi(outputs)
        # "一"的發音的處理 四聲 + 四聲 -> 二聲 + 四聲
        outputs = self.yi_sandhi(outputs)
        # 不是 buˋshiˋ -> 不是 buˊshiˋ
        outputs = self.bu_sandhi(outputs)
        return "".join(outputs)
    
    def apply_third_tone_sandhi(self, syllables):
        result = syllables.copy()
        i = 0
        while i < len(result):
            if TONE_3 in result[i]:
                j = i
                # 連続する三声を探す
                while j < len(result) and TONE_3 in result[j]:
                    j += 1
                # i〜j-1 が三声の塊
                for k in range(i, j - 1):
                    result[k] = result[k].replace(TONE_3, TONE_2)
                i = j
            else:
                i += 1
        return result
    
    def get_tone(self, s):
        if "˥˩" in s: return 4
        if "˨˩˦" in s: return 3
        if "˧˥" in s: return 2
        if "˥" in s: return 1
        return 0

    def yi_sandhi(self, pinyins):
        result = pinyins.copy()
        for i in range(len(result) - 1):
            if result[i] == "i˥": #1
                if self.get_tone(result[i+1]) == 4:
                    result[i] = "i˧˥" #2
                else:
                    result[i] = "i˥˩" #4
        return result
    
    def bu_sandhi(self, pinyins):
        result = pinyins.copy()
        for i in range(len(result) - 1):
            if result[i] == "pu˥˩": #4
                if result[i] == "pu˥˩" and self.get_tone(result[i+1]) == 4:
                    result[i] = "pu˧˥" #2
                else:
                    result[i] = "pu˥˩" #4
        return result


if __name__ == "__main__":
    h = G2P_Chinese_to_Phoneme()
    a = h('我不是一個美國人。対不対？')
    print(a)
    #　--> ['wɔ˨˩˦', 'pu˧˥', 'ʂɨ˥˩', 'i˧˥', 'kɤ˥˩', 'meɪ˨˩˦', 'kwɔ˧˥', 'ʐən˧˥', 'pau']
