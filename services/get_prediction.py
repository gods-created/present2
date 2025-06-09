from services.abstract_service import AbstractService
from exceptions import InvalidFileExtension
from os.path import (
    join,
    exists
)
from joblib import load

class GetPrediction(AbstractService):
    def __init__(self):
        pass

    def _preprocess_text(self, text: str) -> str:
        return super()._preprocess_text(text)

    def predict(
        self, 
        human_description: str, 
        celebration: str,
        model_filename: str,
        vectorizer_filename: str
    ) -> str:
        status, err_description, presents = False, None, None

        try:
            if not all((
                model_filename.endswith('.plk'),
                vectorizer_filename.endswith('.plk'),
            )):
                raise InvalidFileExtension(f'The \'{model_filename}\' or \'{vectorizer_filename}\' has invalid extension.')

            model_fullpath = join('ai_settings', 'model', model_filename)
            vectorizer_fullpath = join('ai_settings', 'vectorizer', vectorizer_filename)

            if not all((
                exists(model_fullpath),
                exists(vectorizer_fullpath)
            )):
                raise FileNotFoundError(f'The \'{model_filename}\' or \'{vectorizer_filename}\' didn\'t find.')

            model, vectorizer = load(model_fullpath), load(vectorizer_fullpath)
            human_description = self._preprocess_text(human_description)
            celebration = self._preprocess_text(celebration)
            features = human_description + ' ' + celebration
            X = vectorizer.transform([features])
            presents, *_ = model.predict(X)

            status = not status
                                                 
        except InvalidFileExtension as e:
            err_description = str(e)

        except Exception as e:
            err_description = f'\'predict\' method execute error: \'{str(e)}\''
        
        finally:
            return status, err_description, presents
            