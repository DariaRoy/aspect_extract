from save_in_xml import save_result
from freq import freq
from pos_analisys import pos_sequence
from pos_analisys import lingvistic
from svm import svm
from load import load_text
from pair_extract import pair
from apply_to_hotels import load_hotel


def y_nn():
    with open('y_nn.txt') as f:
        y = list(map(int, f.read().split()))
        return y


if __name__ == '__main__':

    # y_result = y_nn()
    # print(y_result)

    X_pure_train, X_sentences_train, aspects_list_train, _ = load_text('SentiRuEval_rest_markup_train.xml')
    #X_pure_test, X_sentences_test, aspects_list_test, X_p = load_text('SentiRuEval_rest_markup_test.xml')
    X_pure_test, X_sentences_test, X_p = load_hotel()

    y_result1, y_result2, y_result3 = svm(X_pure_train, X_sentences_train,aspects_list_train, X_pure_test, X_sentences_test)

    pair(X_p, y_result3)

    #y_result = lingvistic(X_pure_test)
    #y_result = freq(X_pure_train, X_pure_test)

    #print (len(X_pure_test), len(y_result1))

    #save_result(X_pure_test, y_result, 'SentiRuEval_result_rest_test_on_rest_2LSTM.xml')
    # save_result(X_pure_test, y_result1, 'SentiRuEval_result_rest_test_on_rest_CNN.xml')
    # save_result(X_pure_test, y_result2, 'SentiRuEval_result_rest_test_on_rest_RF_myw2v.xml')
    # save_result(X_pure_test, y_result3, 'SentiRuEval_result_rest_test_on_rest_GNB_myw2v.xml')