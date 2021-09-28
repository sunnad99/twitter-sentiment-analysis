# Essential libraries for a web app
from flask import Flask, render_template, url_for
from forms import PredictForm
from initialization import keras_tokenizer, sentiment_model, preprocess_data

import keras
from keras.preprocessing.sequence import pad_sequences

# Keras Tokenizer Arguments
TRUNC_TYPE = "post"
PAD_TYPE = "post"
MAX_LEN = 50



# Creating the web app
sentiment_app = Flask(__name__)
sentiment_app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


@sentiment_app.route('/result')
def result():

    return render_template('result.html')


@sentiment_app.route('/',methods=["GET","POST"])
def predict():

    predict_form = PredictForm()

    if predict_form.validate_on_submit():

        processed_tweet = [preprocess_data(predict_form.tweet.data)]
        tokenized_tweet = keras_tokenizer.texts_to_sequences(processed_tweet)
        sequenced_tweet = pad_sequences(tokenized_tweet, padding = PAD_TYPE, truncating = TRUNC_TYPE, maxlen = MAX_LEN)
        predicted_output = sentiment_model.predict(sequenced_tweet)


        type_of_sentiment =  int(keras.backend.argmax(predicted_output))
        percentage_finalized = round((predicted_output[0][type_of_sentiment] * 100))

        
        return render_template('result.html', data = [type_of_sentiment, percentage_finalized])

    return render_template('predict.html', title = 'Predict', form = predict_form)


if __name__ == "__main__":
    sentiment_app.run()
