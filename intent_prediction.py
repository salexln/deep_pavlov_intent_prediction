from deeppavlov import build_model, configs
# from tensorflow.keras import backend
import argparse

from tensorflow.python.framework import ops
ops.reset_default_graph()

# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"

# CONFIG_PATH = configs.classifiers.intents_snips  # could also be configuration dictionary or string path or `pathlib.Path` instance
#
# model = build_model(CONFIG_PATH, download=True)  # in case of necessity to download some data
#
# model = build_model(CONFIG_PATH, download=False)  # otherwise
#
# print(model(["What is the weather in Boston today?"]))

def prepare_data():
    CONFIG_PATH = configs.classifiers.intents_snips  # could also be configuration dictionary or string path or `pathlib.Path` instance
    model = build_model(CONFIG_PATH, download=True)  # in case of necessity to download some data
    model = build_model(CONFIG_PATH, download=False)  # otherwise

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
    # print(model(["What is the weather in Boston today?"]))

# def run(arguments):
def run():
    # if arguments.prepare_data:
    model=prepare_data()
    # if model and arguments.run_test:
    run_test(model=model)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DeepPavlov example')

    # parser.add_argument('--prepare-data',
    #                     action='store_true',
    #                     help='Downloads the data and prepares the model. \
    #                     First time it could take some time')
    #
    # parser.add_argument('--run-test',
    #                     action='store_true',
    #                     help='Test DeepPavlov intent prediction')

    # run(arguments=parser.parse_args())
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
