
import googletrans
from googletrans import Translator


def lang_translate(sentence):
    trans = Translator()
    t = trans.translate(sentence)
    translated_text = {
        "source_lang": t.src,
        "original_text": t.origin,
        "final_text": t.text
    }
    print(f'Language: {translated_text["source_lang"]}')
    if t.src != 'en':
        print(f'Text: {translated_text["original_text"]}')
        print(f'Translated Text: {translated_text["final_text"]}')
    return translated_text
