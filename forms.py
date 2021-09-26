from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length



class PredictForm(FlaskForm):

    tweet = StringField(label = 'Enter the desired tweet', validators = [DataRequired()])
    
    submit = SubmitField('Predict Sentiment')