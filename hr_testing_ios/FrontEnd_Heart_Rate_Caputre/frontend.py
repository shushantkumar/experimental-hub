from flask import Flask
import datetime

app = Flask(__name__)


@app.route("/submit_heart_rate/<heart_rate>")
def submit_heart_rate(heart_rate):
    print(print_to_server(heart_rate))
    return heart_rate


def print_to_server(heart_rate):
    now = datetime.datetime.now()
    print(now, heart_rate)


if __name__ == "__main__":
    app.run(host="192.168.0.138", port=5002, debug=False)
