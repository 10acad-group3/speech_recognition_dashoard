import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TokenizerWrap(Tokenizer):
    def __init__(self, texts, padding, len_sent, filters, reverse=False):
        Tokenizer.__init__(self, filters=filters, char_level=True)

        self.len_sent = len_sent
        self.fit_on_texts(texts)

        self.index_to_word = dict(
            zip(self.word_index.values(), self.word_index.keys()))
        self.tokens = self.texts_to_sequences(texts)

        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre'
        else:
            truncating = 'post'

        self.tokens_padded = pad_sequences(self.tokens,
                                           maxlen=len_sent,
                                           padding=padding,
                                           truncating=truncating
                                           )

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = "".join(words)
        return text

    def text_to_tokens(self, text, reverse=False, padding=False):
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            tokens = np.flip(tokens, axis=1)
            truncating = 'pre'
        else:
            truncating = 'post'

        if padding:
            tokens = pad_sequences(tokens,
                                   maxlen=self.len_sent,
                                   padding=truncating,
                                   truncating=truncating
                                   )
        return tokens
