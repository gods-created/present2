from services import (
    ModelTraining,
    GetPrediction
)
from unittest import TestCase
from os import remove
from csv import writer
from os.path import join
from loguru import logger

class Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_filename = None 
        cls.vectorizer_filename = None

    def setUp(self):
        self.csvfile_name = 'data.csv'
        with open(self.csvfile_name, mode='w') as f:
            csvwriter = writer(f)
            csvwriter.writerows([
                ['human_description', 'celebration', 'presents'],

                ['teenager, plays video games and does sports', 'Wedding', 'hotel voucher, glass set, blanket, decoration, painting'],
                ['retired person, interested in history and loves reading', 'Birthday', 'watch, bracelet'],
                ['adult man, works in IT, likes board games', 'New Year', 'candles, Christmas tree, champagne, pajamas, New Year toy'],
                ['10-year-old child, loves drawing and building blocks', 'Christmas', 'box of sweets, postcard, warm socks, Christmas ornament'],
                ['young couple, recently moved into a new apartment', 'Christmas', 'Christmas ornament, scarf, Christmas wreath, postcard'],
                ['adult man, works in IT, likes board games', 'Christmas', 'Christmas ornament, sweater, postcard, warm socks, box of sweets'],
                ['adult man, works in IT, likes board games', 'Christmas', 'Christmas ornament, warm socks, box of sweets'],
                ['young woman, interested in yoga and travel', 'Other', 'phone case, air freshener, USB flash drive'],
                ['student, interested in movies and music', 'Christmas', 'warm socks, Christmas ornament, Christmas wreath'],
                ['elderly woman, enjoys gardening and embroidery', 'Wedding', 'painting, decoration, dinnerware set'],
                ['adult man, works in IT, likes board games', 'Wedding', 'blanket, glass set, photo frame'],
                ['10-year-old child, loves drawing and building blocks', 'Other', 'phone case, air freshener, notebook, flowers, pen'],
            ])

    def test_1_model_training(self):
        obj = ModelTraining()
        response = obj.train(
            self.csvfile_name,
            None,
            None
        )

        status, err_description, *_ = response
        if not status:
            logger.error(f'\'test_1_model_training\' execute error: \'{err_description}\'')

        self.assertTrue(status)
        self.assertIsNone(err_description)

        _, _, model_filename, vectorizer_filename = response
        Tests.model_filename, Tests.vectorizer_filename = (
            model_filename, 
            vectorizer_filename
        )

    def test_2_get_prediction(self):
        obj = GetPrediction()
        response = obj.predict(
            'retired person, interested in history and loves reading', 'Other',
            Tests.model_filename,
            Tests.vectorizer_filename
        )

        status, err_description, presents = response
        if not status:
            logger.error(f'\'test_2_get_prediction\' execute error: \'{err_description}\'')

        self.assertTrue(status)
        self.assertIsNone(err_description)
        self.assertIsInstance(presents, str)

    def tearDown(self):
        remove(self.csvfile_name)

    @classmethod
    def tearDownClass(cls):
        remove(
            join('ai_settings', 'model', cls.model_filename)
        )

        remove(
            join('ai_settings', 'vectorizer', cls.vectorizer_filename)
        )
    