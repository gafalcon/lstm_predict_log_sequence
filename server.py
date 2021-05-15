from flask import Flask, jsonify, request, render_template, Response
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename
from bson import ObjectId
import predict_server as predict
import requests
import traceback
from datetime import datetime
from db.db import initialize_db
from db.models import Seq, Action
from utils import normalize_url, getObjectType, parse_log_file

app = Flask(__name__)
CORS(app)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost/predictKF'
}
initialize_db(app)
socketio = SocketIO(app, cors_allowed_origins="*")
predict = predict.Prediction("lstmv1", "seq2seq")
predict.initialize_seq()
timestamp = None
currentSeqId = None
uploadLogDir = './uploaded_logs'

@app.route('/sequences')
def get_sequences():
    sequences = Seq.objects().to_json()
    return Response(sequences, mimetype="application/json", status=200)

@app.route('/sequences', methods=['POST'])
def post_sequence():
    seq = Seq(user='gabo')
    seq.save()
    return Response(seq.to_json(), mimetype='application/json', status=200)

@app.route('/init')
def init_seq():
    global timestamp, currentSeqId
    predict.initialize_seq()
    action, top_5_pred, top_5_p = predict.predict_next_action()

    response = {'action': 'enterView',
                'pred': list(top_5_pred),
                'p': list(top_5_p.astype(float))}

    new_seq = Seq(user='gabo',
                  actions=[Action(url='/api/view/objectId',
                                  norm_url='/api/view/objectId',
                                  method='GET',
                                  timestamp=datetime.utcnow)],
                  predictions=[top_5_pred[0]],
                  prediction_probs=[top_5_p[0]])
    new_seq.save()
    currentSeqId = new_seq.id
    socketio.emit('first_action', response)
    timestamp = None
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict_next_action():
    if request.method == 'POST':
        seq_type = request.json.get('seq_type', None)
        action, top_5_pred, top_5_p = predict.predict_next_action(seq_type)
        response = {'action': action,
                    'pred': list(top_5_pred),
                    'p': list(top_5_p.astype(float))}

        socketio.emit('new_action', response)
        return jsonify(response)


@app.route('/predictNext', methods=['POST'])
def predict_n():
    global timestamp
    if timestamp is None:
        timestamp = request.json["timestamp"]
        delta = 0.0
    else:
        delta = (request.json["timestamp"] - timestamp)/ 1000.0
        if (delta > 1 or request.json['req_type'] != 'GET'):
            timestamp = request.json["timestamp"]
        else:
            print("Ignore")
            return jsonify({"res": 'ignored'})
    url = request.json['url']
    normalized_url = normalize_url(url)
    req = request.json['req_type'] + ' ' + normalized_url
    if (request.json.get('contribType') is not None):
        objectId = request.json.get('contribType')
    else:
        objectId = getObjectType(normalized_url, url)
    action, top_5_pred, top_5_p = predict.predict_next_action(
        None, request.json['req_type'], normalized_url, delta, objectId)

    if (currentSeqId):
        Seq.objects(id=currentSeqId).update(
            push__actions=Action(url=url,
                                 norm_url=normalized_url,
                                 method=request.json['req_type'],
                                 objType= objectId),
            push__predictions=top_5_pred[0],
            push__prediction_probs=top_5_p[0]
        )
    response = {'action': f"{req} {objectId}",
                'pred': list(top_5_pred),
                'p': list(top_5_p.astype(float))}

    socketio.emit('new_action', response)
    return jsonify({"res": 'OK'})

@app.route('/predictSequence', methods=['POST'])
def predictSequence():
    if not 'logfile' in request.files:
        return jsonify({"error": 'no file'})
    logfile = request.files['logfile']
    savefilepath = f"{uploadLogDir}/{secure_filename(logfile.filename)}"
    logfile.save(savefilepath)
    try:
        seq_array, urls, req_types = parse_log_file(savefilepath)
        predictions, probs = predict.predict_sequence(seq_array)
        response = {'urls': urls,
                    'req_types': req_types,
                    'predictions': predictions,
                    'probs': probs}
        # socketio.emit('new_sequence', response)
        return jsonify(response)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return jsonify({'error': 'cannot parse uploaded file'})
    return jsonify({'res': 'ok'})

if __name__ == '__main__':
    socketio.run(app)
