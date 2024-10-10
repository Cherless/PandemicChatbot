from flask import Flask
from flask import request

app = Flask('main')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        template = """
        <form method='POST'>
        <input type='text' placeholder='Username...'>
        <input type='password' placeholder='Password...'>
        <input type='submit' value='Auth'>
        </form>
        """
        return template

    elif request.method == 'POST':
        return 'Wow! Great, you logged in!'
app.run()