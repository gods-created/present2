import nltk
import pandas as pd
from random import choice
from string import (
    ascii_letters,
    digits
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from exceptions import (
    InvalidFileExtension,
    InvalidFileContent,
)
from os.path import (
    exists,
    join
)
from joblib import (
    load,
    dump
)
from services.abstract_service import AbstractService
from loguru import logger

class ModelTraining(AbstractService):
    def __init__(self, *args, **kwargs):
        pass

    def _generate_file_name(self) -> str:
        return ''.join(choice(ascii_letters + digits) for _ in range(25)) + '.plk'
    
    def _preprocess_text(self, text: str) -> str:
        return super()._preprocess_text(text)

    def train(
        self, 
        csv_filename: str, 
        model_filename: str = None, 
        vectorizer_filename: str = None
    ) -> tuple:
        status, err_description = False, None 

        try:
            if not exists(csv_filename):
                raise FileNotFoundError(f'The \'{csv_filename}\' didn\'t find.')
            
            if not csv_filename.endswith('.csv'):
                raise InvalidFileExtension(f'The \'{csv_filename}\' has invalid extension.')

            df = pd.read_csv(csv_filename)
            cols = set(col.lower() for col in df.columns)
            if not {'human_description', 'celebration', 'presents'}.issubset(cols):
                raise InvalidFileContent(f'The \'{csv_filename}\' has invalid content (incorrect table columns).')

            model_fullpath = join('ai_settings', 'model')
            vectorizer_fullpath = join('ai_settings', 'vectorizer')
            if model_filename and vectorizer_filename:
                if not all((
                    model_filename.endswith('.plk'),
                    vectorizer_filename.endswith('.plk'),
                )):
                   raise InvalidFileExtension(f'The \'{model_filename}\' or \'{vectorizer_filename}\' has invalid extension.')

                model_fullpath += f'/{model_filename}'
                vectorizer_fullpath += f'/{vectorizer_filename}'

                if not all((
                    exists(model_fullpath),
                    exists(vectorizer_fullpath)
                )):
                    raise FileNotFoundError(f'The \'{model_filename}\' or \'{vectorizer_filename}\' didn\'t find.')
                
                model, vectorizer = load(model_fullpath), load(vectorizer_fullpath)

            else:
                model_filename, vectorizer_filename = (
                    self._generate_file_name(),
                    self._generate_file_name()
                )

                model_fullpath += f'/{model_filename}'
                vectorizer_fullpath += f'/{vectorizer_filename}'

                model, vectorizer = (
                    MultinomialNB(),
                    CountVectorizer()
                )

            try:
                nltk.download('stopwords')
                nltk.download('punkt_tab')
            except:
                logger.warning(
                    'Application can\'t connection to NLTK server and download \'stopwords\' and \'punkt_tab\' packages.'
                )

            df['cleaned_human_description'] = df['human_description'].apply(self._preprocess_text)
            df['cleaned_celebration'] = df['celebration'].apply(self._preprocess_text)
            df['features'] = df['cleaned_human_description'] + ' ' + df['cleaned_celebration']
            
            X = vectorizer.fit_transform(df['features'])
            Y = df['presents']

            X_train, _, y_train, _ = train_test_split(
                X, Y,
                test_size=.2,
                random_state=42
            )

            model.fit(X_train, y_train)

            dump(model, model_fullpath)
            dump(vectorizer, vectorizer_fullpath)

            status = not status

        except (
            InvalidFileExtension,
            InvalidFileContent,
        ) as e:
            err_description = str(e)

        except Exception as e:
            err_description = f'\'train\' method execute error: \'{str(e)}\''
        
        finally:
            if status:
                return status, err_description, model_filename, vectorizer_filename
            return status, err_description
            
