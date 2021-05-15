import argparse

def parseArgs():
    global model, lr, epochs, save, num_layers, hidden_size
    global dataset_folder, train_data, test_data, all_data
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", help="model to train")
    parser.add_argument('-e', "--epochs", type=int)
    parser.add_argument('-lr', "--learning-rate", type=float)
    parser.add_argument('-s', "--save", help="filename to save model")
    parser.add_argument('-l', "--num-layers", type=int, help="num layers of model")
    parser.add_argument('-hs', "--hidden-size", type=int, help="hidden dimension of rnn model")
    parser.add_argument('-d', "--dataset-folder", help="path to dataset folder")
    args = parser.parse_args()
    if args.model:
        model = args.model
    if args.epochs:
        epochs = args.epochs
    if args.learning_rate:
        lr = args.learning_rate
    if args.save:
        save = args.save
    if args.num_layers:
        num_layers = args.num_layers
    if args.hidden_size:
        hidden_size = args.hidden_size
    if args.dataset_folder:
        dataset_folder = args.dataset_folder
        train_data = f"{dataset_folder}/processed/train/*csv"
        test_data = f"{dataset_folder}/processed/test/*csv"
        all_data = f"{dataset_folder}/processed/*csv"
    print(args)

model = "lstm1"
MODEL_PATH = "./models"
num_layers = 2
lstm_hidden_size = 128
hidden_size = 128
batch_size = 1
seq_len = 30
lr = 0.001
epochs = 50
save = None

dataset_folder = "./datasets/kf/"
train_data = f"{dataset_folder}/processed/train/*csv"
test_data = f"{dataset_folder}/processed/test/*csv"
all_data = f"{dataset_folder}/processed/*csv"

x_columns = ['t_0', 't_1', 't_2','t_3', 'url_0', 'url_1', 'url_2', 'url_3', 'url_4', 'url_5', 'url_6',
             'url_7', 'url_8', 'url_9', 'url_10', 'url_11', 'url_12', 'url_13',
             'url_14', 'url_15', 'url_16', 'url_17', 'delta', 'objTypeId']

y_columns = ['seq_type_0', 'seq_type_1','seq_type_2', 'seq_type_3', 'seq_type_4', 'seq_type_5', 'seq_type_6',
             'seq_type_7', 'seq_type_8', 'seq_type_9', 'seq_type_10', 'seq_type_11','seq_type_12', 'seq_type_13',
             'seq_type_14', 'seq_type_15','seq_type_16', 'seq_type_17', 'seq_type_18', 'seq_type_19', 'seq_type_20',
             'seq_type_21', 'seq_type_22', 'seq_type_23']

input_size = len(x_columns)
output_size = len(y_columns)

type_to_cat = {'GET': [1.0,0.0,0.0,0.0], 'POST': [0.0,1.0,0.0,0.0], 'PUT': [0.0,0.0,1.0,0.0], 'DELETE': [0.0,0.0,0.0,1.0]}
cat_to_request_type = ['GET', 'POST', 'PUT', 'DELETE']
urls = [
    'start',
    '/api/notifications/notify/objectId',
    '/api/communities/objectId/groups',
    '/api/links/from/objectId',
    '/api/links/objectId/search',
    '/api/links/either/objectId',
    '/api/records/object/objectId',
    '/api/records/read/objectId/objectId',
    '/api/links',
    '/api/contributions/objectId',
    '/api/links/',
    '/api/upload',
    '/api/objects/objectId',
    '/api/communities/objectId',
    '/api/authors/objectId/me',
    '/api/links/objectId',
    '/api/objects/objectId/objectId',
    '/api/links/to/objectId',
    'end'
]

seq_types = [
    'start',
    'get_note',
    'get_links_from',
    'record_note_read',
    'get_links_from_contrib',
    'edit_note',
    'notify_comm',
    'record_note_edited',
    'get_object',
    'get_groups',
    'get_links_to_note',
    'search',
    'post_scaffold',
    'get_community',
    'get_note_records',
    'get_author',
    'delete_scaffold',
    'new_note',
    'post_link_view_note',
    'new_attachment',
    'upload_attachment',
    'edit_object',
    'post_link',
    'end'
]

new_row = {'type': 'GET', 'url': 'start', 'delta': 0.0, 'seq_type': 'start'}

seq_type_to_url = {
    'start': 'start',
    'get_note': '/api/objects/objectId',
    'get_links_from': '/api/links/from/objectId',
    'record_note_read': '/api/records/read/objectId/objectId',
    'get_links_from_contrib': '/api/links/from/objectId',
    'edit_note': '/api/objects/objectId',
    'notify_comm': '/api/notifications/notify/objectId',
    'record_note_edited': '/api/records/read/objectId/objectId',
    'get_object': '/api/objects/objectId',
    'get_groups': '/api/communities/objectId/groups',
    'get_links_to_note': '/api/links/to/objectId',
    'search': '/api/links/objectId/search',
    'post_scaffold': '/api/links',
    'get_community': '/api/communities/objectId',
    'get_note_records': '/api/records/object/objectId',
    'get_author': '/api/authors/objectId/me',
    'delete_scaffold': '/api/links/objectId',
    'new_note': '/api/contributions/objectId',
    'post_link_view_note': '/api/links/',
    'new_attachment': '/api/contributions/objectId',
    'upload_attachment': '/api/upload',
    'edit_object':'/api/objects/objectId',
    'post_link': '/api/links',
    'end': 'end'
}

seq_type_to_req_type = {
    'start': 'GET',
    'get_note': 'GET',
    'get_links_from': 'GET',
    'record_note_read': 'POST',
    'get_links_from_contrib': 'GET',
    'edit_note': 'PUT',
    'notify_comm': 'POST',
    'record_note_edited': 'POST',
    'get_object': 'GET',
    'get_groups': 'GET',
    'get_links_to_note': 'GET',
    'search': 'POST',
    'post_scaffold': 'POST',
    'get_community': 'GET',
    'get_note_records': 'GET',
    'get_author': 'GET',
    'delete_scaffold': 'DELETE',
    'new_note': 'POST',
    'post_link_view_note': 'POST',
    'new_attachment': 'POST',
    'upload_attachment': 'POST',
    'edit_object': 'GET',
    'post_link': 'POST',
    'end': 'GET'
}

id_to_objType = ['', 'Note', 'Scaffold', 'Attachment', 'Context']
objType_to_id = {t:i for i,t in enumerate(id_to_objType)}

predicted_seq_types = [
    'start',
    'get_note',
    'record_note_read',
    'edit_note',
    'notify_comm',
    'record_note_edited',
    'get_groups',
    'search',
    'post_scaffold',
    'get_note_records',
    'get_author',
    'delete_scaffold',
    'new_note',
    'new_attachment',
    'edit_object',
    'end'
]

