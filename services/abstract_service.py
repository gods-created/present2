from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from abc import ABC, abstractmethod

class AbstractService(ABC):
    @abstractmethod
    def _preprocess_text(self, text: str) -> str:
        text = text \
            .lower() \
            .translate(
                str.maketrans('', '', punctuation)
            )
        
        stop_words = set(stopwords.words('english'))
        tokens = [
            word for word in word_tokenize(text) if word not in stop_words
        ]

        return ' '.join(tokens)