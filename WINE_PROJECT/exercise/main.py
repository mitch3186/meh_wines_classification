from flask import Flask, request, render_template
from make_prediction import good_or_meh_wine

# create a flask object
app = Flask(__name__, static_url_path='/static')

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/good_or_meh_wine/', methods=['GET', 'POST'])
def render_message():

    # user-entered attributes
    attributes = ['fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides','sulphates', 'alcohol', 'sul_diox_ratio']

    # error messages to ensure correct units of measure
    # input the max amounts that can be entered by user
    messages = ["The amount of fixed acidity must be less than 16.0",
                "The amount of volatile acidity must be less than 2.000.",
                "The amount of residual sugar must be less than 75.00",
                "The amount of chlorides must be less than .750.",
                "The amount of sulphates must be less than 2.50.",
                "The amount of alcohol must be less than 20.0."
                "The ratio of total sulfur dioxide over free sulfur dioxide must be less than 50.0"]

    # hold all attributes as floats
    amounts = []

    # takes user input and ensures it can be turned into a floats
    for i, ing in enumerate(attributes):
        user_input = request.form[ing]
        try:
            float_attribute = float(user_input)
        except:
            return render_template('index.html', message=messages[i])
        amounts.append(float_attribute)

    # show user final message
    final_message = good_or_meh_wine(amounts)
    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    app.run(debug=True)