import streamlit as st
from services.get_prediction import GetPrediction
from dotenv import load_dotenv
from os import getenv

st.set_page_config('Present2', '🎁')
st.title('Present2: AI solution if you don\'t know what to give as a gift')
st.write('''
   Use this AI agent, which will suggest the best gift based on a description of the person and the upcoming holiday      
''')

human_description = st.text_input(
    label='Human description',
    max_chars=250,
    placeholder='teenager, plays video games and does sports'
)

celebration = st.radio(
    label='Celebration',
    options=('New Year', 'Birthday', 'Wedding', 'Christmas', 'Other')
)

if st.button('Get prediction') and all((human_description, celebration)):
    load_dotenv()

    model_filename = getenv('MODEL_FILENAME')
    vectorizer_filename = getenv('VECTORIZER_FILENAME')

    if all((model_filename, vectorizer_filename)):
        obj = GetPrediction()
        status, err_description, presents = obj.predict(
            human_description=human_description,
            celebration=celebration,
            model_filename=model_filename,
            vectorizer_filename=vectorizer_filename
        )

        if status:
            st.success(f'Present list: {presents}.')
        else:
            st.error(err_description)

    elif not model_filename:
        st.error('No trained AI model found to fulfill the request.')

    elif not vectorizer_filename:
        st.error('No trained AI vectorizer found for converting incoming text.')

    else:
        st.warning('Something went wrong. Try it again.')

# python -m streamlit run app.py --server.port=8001 --server.address=0.0.0.0