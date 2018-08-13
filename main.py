#!/usr/bin/env python
import warnings
warnings.simplefilter("ignore")
from keras import Model
from keras.layers import Dense

from keras_pandas.Automater import Automater
from keras_pandas.lib import load_titanic


def main():
    observations = load_titanic()

    # Transform the data set, using keras_pandas
    categorical_vars = ['pclass', 'sex', 'survived']
    numerical_vars = ['age', 'siblings_spouses_aboard', 'parents_children_aboard', 'fare']
    text_vars = ['name']

    auto = Automater(categorical_vars=categorical_vars, numerical_vars=numerical_vars, text_vars=text_vars, response_var='survived')
    X, y = auto.fit_transform(observations)

    # Start model with provided input nub
    x = auto.input_nub

    # Fill in your own hidden layers
    x = Dense(256)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256)(x)

    # End model with provided output nub
    x = auto.output_nub(x)

    model = Model(inputs=auto.input_layers, outputs=x)
    model.compile(optimizer='Adam', loss=auto.loss, metrics=['accuracy'])

    # Train model
    model.fit(X, y, epochs=15, validation_split=.2)

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)


if __name__ == '__main__':
    main()
