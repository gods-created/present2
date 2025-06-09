# Present2

Present2: AI solution if you don\'t know what to give as a gift

## Description

Use this AI agent, which will suggest the best gift based on a description of the person and the upcoming holiday   

## Installation

```bash
git clone https://github.com/gods-created/present2.git
cd present2
pip install -r requirements.txt
```

## Usage

Create a .env file based on .env.example (find ready file names in ai_settings/).

```bash
python manager.py model_training --csv_filename=train_dataset.csv
python -m streamlit run app.py --server.port=8001 --server.address=0.0.0.0
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License