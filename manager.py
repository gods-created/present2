import click
from typing import Optional
from services.model_training import ModelTraining
from services.get_prediction import GetPrediction
from loguru import logger

@click.group()
def cli():
    pass

@cli.command('model_training')
@click.option('--csv_filename', required=True, default=None)
@click.option('--model_filename', default=None)
@click.option('--vectorizer_filename', default=None)
def model_training(
    csv_filename: Optional[str], 
    model_filename: Optional[str], 
    vectorizer_filename: Optional[str]
) -> Optional[str]:    
    obj = ModelTraining()
    response = obj.train(csv_filename, model_filename, vectorizer_filename)
    if len(response) == 2:
        status, err_description = response
        if not status:
            return click.echo(err_description)
        logger.warning(
            'The \'ModelTraining.train\' method return STATUS=TRUE, but response hasn\'t model and vectorizer file names.'
        )
        return None 
    
    _, _, model_filename, vectorizer_filename = response
    logger.success(
        f'Model filename: \'{model_filename}\'. ' \
        f'Vectorizer filename: \'{vectorizer_filename}\''
    )
    return None

@cli.command('get_prediction')
@click.option('--human_description', required=True, default=None)
@click.option('--celebration', required=True, default=None)
@click.option('--model_filename', required=True, default=None)
@click.option('--vectorizer_filename', required=True, default=None)
def get_prediction(
    human_description: Optional[str], 
    celebration: Optional[str], 
    model_filename: Optional[str], 
    vectorizer_filename: Optional[str], 
) -> None:
    obj = GetPrediction()
    status, err_description, presents = obj.predict(
        human_description,
        celebration,
        model_filename,
        vectorizer_filename 
    )

    if status:
        click.echo(presents)
    else:
        click.echo(err_description)

    return None

if __name__ == '__main__':
    cli()

# python manager.py model_training --csv_filename=train_dataset.csv

# python manager.py get_prediction 
# --human_description="middle-aged woman, works as a teacher and loves cooking" 
# --celebration="New Year"
# --model_filename=W8Vnz2kcixVQCgcQQQzV537nC.plk 
# --vectorizer_filename=UPlfWJmjvbw8POgDGbngZQE9l.plk