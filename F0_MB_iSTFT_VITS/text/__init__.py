from .phonemize import Tokenizer
converter = Tokenizer()

def text_to_sequence(text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  clean_text = converter(text)
  sequence = clean_text
  return sequence


def infer_g2p(text):
  clean_text = converter(text)
  sequence = clean_text
  return sequence


if __name__ == "__main__":
  print(text_to_sequence("吾輩は猫である。"))
  print(infer_g2p("吾輩は猫である。"))