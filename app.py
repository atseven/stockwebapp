import pyrebase
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, Response
import stripe
from datetime import date
import yfinance as yf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from plotly import graph_objs as go
import io
from prophet import Prophet
from prophet.plot import plot_plotly, plot_yearly
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

app = Flask(__name__)       

#Add your own details
config = {
  "apiKey": "AIzaSyDFEJXZnQm1ZJ9eTEXkOBqPrGBmCith4hA",
  "authDomain": "stock-web-app-6d13d.firebaseapp.com",
  "storageBucket": "stock-web-app-6d13d.appspot.com",
  "messagingSenderId": "319641312941",
  "databaseURL": "https://stock-web-app-6d13d-default-rtdb.firebaseio.com"
}

#initialize firebase
firebase = pyrebase.initialize_app(config)
print("firebase", firebase)
auth = firebase.auth()
db = firebase.database()

#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}


app.config['STRIPE_PUBLIC_KEY'] = 'pk_test_51Lz50TBg3QLugHf96CUi1lQ09tjWSGZ4ZrxzlmnXxGrgMEYk885vC6eherUPuf6wgTEiiR6OoZnNDWBpT516VBDx00XBPJhLkF'
app.config['STRIPE_SECRET_KEY'] = 'sk_test_51Lz50TBg3QLugHf90vB3K9BAfZTWDrqbEZGdg7wJAgYzwJkbRbFai2jmEh3G94sPxVLAvRVeEaZPlZBER1LcZVuf002bKFoAKG'

stripe.api_key = app.config['STRIPE_SECRET_KEY']


#Login
@app.route("/")
def login():
    return render_template("login.html")

#Sign up/ Register
@app.route("/signup")
def signup():
    return render_template("signup.html")

#Welcome page
@app.route("/welcome")
def welcome():
    if person["is_logged_in"] == True:
        print("Welcome.html")
        # return redirect(url_for('welcome'))
        return render_template("index.html")
        # return render_template("welcome.html", email = person["email"], name = person["name"])
    else:
        print("Login.html")
        return redirect(url_for('login'))

#If someone clicks on login, they are redirected to /result
@app.route("/result", methods = ["POST", "GET"])
def result():
    if request.method == "POST":        #Only if data has been posted
        result = request.form           #Get the data
        email = result["email"]
        password = result["pass"]
        try:
            #Try signing in the user with the given information
            user = auth.sign_in_with_email_and_password(email, password)
            #Insert the user data in the global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            #Get the name of the user
            data = db.child("users").get()
            person["name"] = data.val()[person["uid"]]["name"]
            #Redirect to welcome page
            # return redirect(url_for('welcome'))
            return render_template("index.html")
        except:
            #If there is any error, redirect back to login
            return redirect(url_for('login'))
    else:
        if person["is_logged_in"] == True:
            # return redirect(url_for('welcome'))
            return render_template("index.html")
        else:
            return redirect(url_for('login'))

#If someone clicks on register, they are redirected to /register
@app.route("/register", methods = ["POST", "GET"])
def register():
    if request.method == "POST":        #Only listen to POST
        result = request.form           #Get the data submitted
        email = result["email"]
        password = result["pass"]
        name = result["name"]
        print(email, password, name)
        try:
            #Try creating the user account using the provided data
            auth.create_user_with_email_and_password(email, password)
            #Login the user
            user = auth.sign_in_with_email_and_password(email, password)
            #Add data to global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            person["name"] = name
            #Append data to the firebase realtime database
            data = {"name": name, "email": email}
            db.child("users").child(person["uid"]).set(data)
            #Go to welcome page

            print("Something is going in db")
            # return redirect(url_for('welcome'))
            return render_template("index.html")
        except Exception as e:
            print("Error: ", e)
            #If there is any error, redirect to register
            return redirect(url_for('register'))

    else:
        if person["is_logged_in"] == True:
            # return redirect(url_for('welcome'))
            return render_template("index.html")
        else:
            return redirect(url_for('register'))


@app.route('/stripe_pay')
def stripe_pay():
    print("stripe_pay")
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price': 'price_1M08JoBg3QLugHf9KwRc1Kvb',
            'quantity': 1,
        }],
        mode='payment',
        success_url=url_for('thanks', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
        cancel_url=url_for('welcome', _external=True),
    )
    print("Stripe Pay 2")
    return {
        'checkout_session_id': session['id'], 
        'checkout_public_key': app.config['STRIPE_PUBLIC_KEY']
    }

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

@app.route('/stripe_webhook', methods=['POST'])
def stripe_webhook():
    print('WEBHOOK CALLED')

    if request.content_length > 1024 * 1024:
        print('REQUEST TOO BIG')
        abort(400)
    payload = request.get_data()
    sig_header = request.environ.get('HTTP_STRIPE_SIGNATURE')
    endpoint_secret = 'whsec_e9aaad28e52786ccd3a721c6bee5ad2d87486ce9b0f7d4eef43138e33dc9980d'
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        print('INVALID PAYLOAD')
        return {}, 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        print('INVALID SIGNATURE')
        return {}, 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(session)
        line_items = stripe.checkout.Session.list_line_items(session['id'], limit=1)
        print(line_items['data'][0]['description'])

    return {}


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
       TODAY = date.today().strftime("%Y-%m-%d")
       stock_name = request.form.get("symbol")
       START = request.form.get("sdate")
       colName = request.form.get("predcolname")
       days = int(request.form.get("dayspred"))

       def import_data():
            # st.subheader("Last 5 Rows of Dataset:")
            dataframe = yf.Ticker(str(stock_name))
            dataframe=dataframe.history(start=START, end=TODAY, period="max").reset_index()

            # print(type(dataframe['Date'][0]), dataframe['Date'][0])
            dataframe['Date'] = pd.to_datetime(dataframe['Date'])
            dataframe['Date'] = dataframe['Date'].dt.date
            print(str(stock_name) + ": ", dataframe)

            dataframe.drop(dataframe.columns[dataframe.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
            # dataframe = dataframe.sort_values('Date').reset_index(drop=True)
            
            # st.write(dataframe.tail())
            return dataframe
       def plot_data(df):
            # figure = go.Figure()
            # # figure.add_trace(go.Scatter(x=df['Date'], y = df['Open'], name="Predicted " + colName))
            # figure.add_trace(go.Scatter(x=df['Date'], y = df[colName], name="Predicted " + colName))
            # figure.layout.update(title_text="Date vs " + colName, xaxis_rangeslider_visible=True)
            # st.plotly_chart(figure)
            # return
            values = list(df[colName].values)
            labels = [str(i) for i in df['Date']]

            # print("labels: ", labels[:5], type(labels[0]))

            return labels, values
            # fig = Figure()
            # ax = fig.subplots()
            # print("True")
            # ax.plot(df['Date'], df[colName])
            # print("False")
            # # Save it to a temporary buffer.
            # buf = BytesIO()
            # fig.savefig(buf, format="png")
            # # Embed the result in the html output.
            # data = base64.b64encode(buf.getbuffer()).decode("ascii")
            # return f"<img src='data:image/png;base64,{data}'/>"

            # fig = px.bar(df, x='Date', y=colName)
            # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            # return graphJSON
       def prepare_data(df):
            df[colName] = df[colName].astype(float)
            num_shape = df.shape[0] - 500
            window = 60
            # st.write(df.shape)
            # train/test split
            train = df.iloc[:num_shape, 1:2].values
            test = df.iloc[num_shape:, 1:2].values
            df_price = np.vstack((train, test))

            sc = MinMaxScaler(feature_range = (0, 1))

            X_train, Y_train = [], []
            # print(type(train))
            # st.write(train)
            train_scaled = sc.fit_transform(train)

            for i in range(train.shape[0]-window):
                batch = np.reshape(train_scaled[i:i+window, 0], (window, 1))
                X_train.append(batch)
                Y_train.append(train_scaled[i+window, 0])
            X_train = np.stack(X_train)
            Y_train = np.stack(Y_train)

            X_test, Y_test = [], []
            test_scaled = sc.fit_transform(df_price[num_shape-window:])

            for i in range(test.shape[0]):
                batch = np.reshape(test_scaled[i:i+window, 0], (window, 1))
                X_test.append(batch)
                Y_test.append(test_scaled[i+window, 0])

            return X_train, Y_train, X_test, Y_test, sc
       def train(df):
            try:
                dfTrain = df[['Date', colName]]
                dfTrain = dfTrain.rename(columns = {"Date": "ds", colName: "y"})
                # X_train, Y_train, X_test, Y_test, sc = prepare_data(df)	
                # LSTM_model_training(X_train, Y_train, X_test, Y_test, sc)
                # GRU_model_training(X_train, Y_train, X_test, Y_test, sc)
                # st.write("Training completed!")
                # st.write("Please wait for a while, we are displaying output predictions for you & evaluating it...")
                print(True)
                m = Prophet(interval_width=0.95, daily_seasonality=True, yearly_seasonality=True)
                print("Prophet", dfTrain.head())
                m.fit(dfTrain)
                print("fit", days, type(days))
                p = m.make_future_dataframe(periods = days, freq='D')
                f = m.predict(p)
                pDf = f[['ds', 'yhat']]
                pDf = pDf.rename(columns = {"ds": "Date", "yhat": "Predicted Price"})
                # st.write("R2 Score is: " + str(r2_score(list(df[colName]), list(pDf['Predicted Price'].iloc[:-int(days)]))))
                # st.write("RMSE Score is: " + str(sqrt(mean_squared_error(list(df[colName]), list(pDf['Predicted Price'].iloc[:-int(days)])))))

                # st.subheader("Predicted Data")
                # st.write(pDf.tail())
                # st.subheader("Prediction Plot")
                
                # fig = plot_plotly(m, f)
                # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                # return render_template('thanks.html', graphJSON=graphJSON)
                predValues = list(pDf["Predicted Price"].values)
                predLabels = [str(i).split(" ")[0] for i in pDf['Date']]

                print("predLabels: ", predLabels[:5], type(predLabels[0]))
                print("predValues: ", predValues[:5], type(predValues[0]))

                return predLabels, predValues, pDf
            except Exception as e:
                print("ERROR  : ", e)
                return [], []


            
       DF = import_data()
       labels, values = plot_data(DF)
       predLabels, predValues, pDf = train(DF)
    #    return render_template('thanks.html', graphJSON=r)
       return render_template("plots.html", 
                        labels = labels, 
                        values = values, 
                        predLabels=predLabels[-days-1: ], 
                        predValues=predValues[-days-1: ],
                        actual_df_col_names=DF.columns.values, 
                        actual_df_row_data=list(DF.tail().values.tolist()),
                        pred_df_col_names=pDf.columns.values, 
                        pred_df_row_data=list(pDf.tail(days).values.tolist()),
                        zip = zip,
                        predColName = colName,
                        days = days

                        )

    #    return "Your name is"
    else:
        return render_template("thanks.html")
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
