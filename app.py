from flask import Flask, render_template, request, jsonify
from agent_module import Agent

app = Flask(__name__)
agent = Agent()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json.get('user_input')
    image_path = request.json.get('image_path')  # 可选的图像路径参数
    response = agent.handle_query(user_input, image_path)
    return jsonify({'response': response})


@app.route('/get_dialog_history', methods=['GET'])
def get_dialog_history():
    history = agent.get_dialog_history()
    return jsonify(history)


if __name__ == '__main__':
    app.run(debug=False)
