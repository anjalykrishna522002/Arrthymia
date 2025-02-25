from flask import Flask
from public import public
# from user import user

app=Flask(__name__)
app.secret_key="value"
app.register_blueprint(public)
# app.register_blueprint(user)
app.run(debug=True,port=5169)