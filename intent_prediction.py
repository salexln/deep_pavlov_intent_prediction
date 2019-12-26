from deeppavlov import build_model, configs
import argparse



def prepare_data():
    CONFIG_PATH = configs.classifiers.intents_snips
    model = build_model(CONFIG_PATH, download=True)
    model = build_model(CONFIG_PATH, download=False)

    return model

def run_test(model):
    """
    The trained data set supports the following intents:
        - GetWeather
        - BookRestaurant
        - PlayMusic
        - AddToPlaylist
        - RateBook
        - SearchScreeningEvent
        - SearchCreativeWork
    """
    phrases_to_test = ['What is the weather today in Tel Aviv?',
                       'Play me Tylor swift',
                       'What do you think of Moby Dick?',
                       'Can you book me a place in Pizza hut?']

    for phrase in phrases_to_test:
        print(f"{phrase}:{model([phrase])}")

def run():
    model=prepare_data()
    run_test(model=model)

if __name__ == '__main__':
    run()

"""
Preparations:
Based on: http://docs.deeppavlov.ai/en/master/features/models/classifiers.html
1. Create virutal env: python3 -m venv env
2. Activate: source ./env/bin/activate
3. Install DeepPavlov: pip3 install deeppavlov
4. Install FastText: python3 -m pip install pybind11==2.2.3 git+https://github.com/deepmipt/fastText.git#egg=fastText==0.8.22
5. Install TensorFlow: pip3 install --ignore-installed --upgrade tensorflow
"""
