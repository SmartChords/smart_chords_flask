from sklearn.utils import shuffle
import csv


def find_key_signature(notes_list):
    key_sig = [i for i in notes_list if "key-" in i]
    if (len(key_sig) > 0):
        key_value = key_sig[0].split("-")[1]
        return key_value
    else:
        return "0.001"

def separate_bars_of_notes(notes_list):
    size = len(notes_list)
    idx_list = [idx + 1 for idx, val in enumerate(notes_list) if val == 'BAR']
    list_of_notes = [notes_list[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]

    for n in list_of_notes:
        if 'BAR' in n:
            n.remove('BAR')

    return list_of_notes

def build_model_input(notes_list):
    key = find_key_signature(notes_list)
    new_notes_list = notes_list[:]
    new_notes_list.remove(key)
    separated_list = separate_bars_of_notes(new_notes_list)
    input_setup = set_notes_values(separated_list)
    for i in input_setup:
        i.append(key)

    write_inputs_to_csv(input_setup)

def pad_or_truncate(list, target_len):
    unused_space = target_len - len(list)
    return list[:target_len] + [shuffle(list)[0] for _ in range(unused_space)]

def unique(list):
    unique_list = []

    for x in list:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def split_long_measures(arr, ref_array):
    if len(arr) > 4:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]

        split_long_measures(L,ref_array)
        split_long_measures(R,ref_array)

    else:
        ref_array.append(pad_or_truncate(arr,5))


def set_notes_values(notes_list):
    note_inputs = []
    for n in notes_list:
        if len(n) <= 3:
            note_inputs.append(pad_or_truncate(n,5))
        elif len(n) == 4:
            uniq = unique(n)
            note_inputs.append(pad_or_truncate(uniq,5))
        else:
            split_long_measures(n,note_inputs)

    return note_inputs


def write_inputs_to_csv(prepared_input):
    fields = ['note1', 'note2', 'note3', 'not4', 'note5', 'key']
    rows = prepared_input
    filename = "input.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)


def get_chord_predictions():
    reconstructed_model = keras.models.load_model("chord_model")
    data = pd.read_csv("chord_data/input.csv")

    prediction = np.array(data)
    predictions = np.argmax(reconstructed_model.predict(prediction), axis=-1)
    prediction_ = np.argmax(to_categorical(predictions), axis = 1)
    prediction_ = encoder.inverse_transform(prediction_)
    return prdiction_
