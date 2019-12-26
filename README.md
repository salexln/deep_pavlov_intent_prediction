# Simple intent prediction with Deep Pavlov

(based on http://docs.deeppavlov.ai/en/master/features/models/classifiers.html)
# Installations:
1. Create virutal env: `python3 -m venv env`
2. Activate: `source ./env/bin/activate`
3. Install DeepPavlov: `pip3 install deeppavlov`
4. Install FastText: `python3 -m pip install pybind11==2.2.3 git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22`
5. Install TensorFlow: `pip3 install --ignore-installed --upgrade tensorflow`

# How to run:
Run `python3 intent_prediction.py`

Notes:
1. Running the first time can take ~ an hour because DP downloads a lot of data
2. This mode supports the following intents:
    - GetWeather
    - BookRestaurant
    - PlayMusic
    - AddToPlaylist
    - RateBook
    - SearchScreeningEvent
    - SearchCreativeWork
3. The phrases that we test are defined in the code as `phrases_to_test`
